//! Explicit session-bound project/media API. Private snapshots never enter job events.
use super::{contract::*, editing::EditRequest, media, persistence::*, service::Snapshot};
use crate::api::App;
use axum::{
    Json, Router,
    extract::{Path, Request, State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write, path::PathBuf};
#[path = "assembly_api.rs"]
mod assembly;
#[path = "authoring_api.rs"]
mod authoring;
#[path = "library_api.rs"]
mod libraries;
#[path = "runs_api.rs"]
mod runs;

#[path = "captures_api.rs"]
mod captures;
#[path = "lora_catalog_api.rs"]
mod lora_catalog;
#[path = "media_health_api.rs"]
mod media_health;
#[path = "reference_cache_api.rs"]
mod reference_cache;
#[path = "streaming_import_api.rs"]
mod streaming_import;

pub struct Error {
    status: StatusCode,
    code: String,
    snapshot: Option<Box<Snapshot>>,
}
impl Error {
    fn new(status: StatusCode, code: &str) -> Self {
        Self {
            status,
            code: code.into(),
            snapshot: None,
        }
    }
}
impl From<StorageError> for Error {
    fn from(error: StorageError) -> Self {
        let status = match error.code.as_str() {
            "revision_conflict"
            | "asset_revision_conflict"
            | "storage_conflict"
            | "storage_root_changed"
            | "project_already_open" => StatusCode::CONFLICT,
            "project_not_found" | "take_not_found" | "resource_missing" | "location_not_found" => {
                StatusCode::NOT_FOUND
            }
            _ => StatusCode::UNPROCESSABLE_ENTITY,
        };
        Self::new(status, &error.code)
    }
}
impl From<Issue> for Error {
    fn from(issue: Issue) -> Self {
        StorageError { code: issue.code }.into()
    }
}
impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let mut body = json!({"error":{"code":self.code}});
        if let Some(snapshot) = self.snapshot {
            body["snapshot"] = json!(snapshot);
        }
        (self.status, Json(body)).into_response()
    }
}
fn input<T>(value: Result<Json<T>, JsonRejection>) -> Result<T, Error> {
    value
        .map(|Json(v)| v)
        .map_err(|_| Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request"))
}
fn schema(version: u32) -> Result<(), Error> {
    if version != 1 {
        return Err(Error::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_schema",
        ));
    }
    Ok(())
}
fn internal(_: impl std::fmt::Display) -> Error {
    Error::new(StatusCode::INTERNAL_SERVER_ERROR, "storage_error")
}
fn media_error(code: &'static str) -> Error {
    Error::new(StatusCode::UNPROCESSABLE_ENTITY, code)
}
fn staging(app: &App) -> Result<SessionStaging, Error> {
    let path = app.store.staging_directory().map_err(media_error)?;
    SessionStaging::select(&path).map_err(Into::into)
}
pub fn routes() -> Router<App> {
    Router::new()
        .merge(reference_cache::routes())
        .merge(captures::routes())
        .merge(lora_catalog::routes())
        .merge(media_health::routes())
        .merge(libraries::routes())
        .merge(runs::routes())
        .merge(authoring::routes())
        .merge(assembly::routes())
        .route("/v1/video/capabilities", get(capabilities))
        .route("/v1/video/locations", post(location))
        .route("/v1/video/projects", post(create))
        .route("/v1/video/projects/at-location", post(create_at))
        .route("/v1/video/projects/open", post(open))
        .route("/v1/video/projects/{id}", get(read))
        .route("/v1/video/projects/{id}/edits", post(edit))
        .route(
            "/v1/video/projects/{id}/scenes/{scene}/compiled",
            get(compiled),
        )
        .route(
            "/v1/video/projects/{id}/scenes/{scene}/preflight",
            get(preflight),
        )
        .route("/v1/video/projects/{id}/review", post(review))
        .route("/v1/video/projects/{id}/assets", post(attach))
        .route("/v1/video/projects/{id}/save", post(save))
        .route("/v1/video/projects/{id}/save-as", post(save_as))
        .route(
            "/v1/video/projects/{id}/export-compatible",
            post(export_compatible),
        )
        .route("/v1/video/assets", post(upload))
        .route("/v1/video/assets/{id}", get(asset))
        .route("/v1/video/assets/{id}/waveform", get(waveform))
}
#[derive(Deserialize)]
struct WaveformQuery {
    #[serde(default = "waveform_peaks")]
    peaks: usize,
}
fn waveform_peaks() -> usize {
    96
}
async fn waveform(
    State(app): State<App>,
    Path(id): Path<String>,
    axum::extract::Query(query): axum::extract::Query<WaveformQuery>,
) -> Result<Json<super::waveform::Waveform>, Error> {
    let info = app
        .store
        .asset(&id, "local")
        .await
        .map_err(internal)?
        .ok_or_else(|| Error::new(StatusCode::NOT_FOUND, "asset_not_found"))?;
    let file = app
        .store
        .open_media(&id)
        .map_err(internal)?
        .ok_or_else(|| Error::new(StatusCode::NOT_FOUND, "asset_not_found"))?;
    let mime = info["mime_type"].as_str().unwrap_or("").to_owned();
    // The lease follows the blocking decoder, including if the HTTP caller leaves.
    static LIMIT: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
        std::sync::OnceLock::new();
    let lease = LIMIT
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(2)))
        .clone()
        .acquire_owned()
        .await
        .map_err(internal)?;
    Ok(Json(
        tokio::task::spawn_blocking(move || {
            let _lease = lease;
            super::waveform::decode(&file, &mime, query.peaks)
        })
        .await
        .map_err(internal)?
        .map_err(media_error)?,
    ))
}
async fn capabilities(State(app): State<App>) -> Json<Value> {
    let available = crate::api::h3_available(&app);
    let models = super::adapters::h3::variants()
        .into_iter()
        .map(|(model, mut capabilities)| {
            capabilities.runtime_available = available;
            json!({"model":model,"capabilities":capabilities})
        })
        .collect::<Vec<_>>();
    Json(
        json!({"schema_version":1,"media":{"mime_types":media::MIME_TYPES,"max_upload_bytes":null,"streaming":true,"max_buffered_image_bytes":media::MAX_UPLOAD},
        "project_operations":["create","create_at_location","read","edits","review","attach_asset","save","open","save_as","export_compatible","preflight"],"models":models,"authoring":super::authoring::steps::limits(),
        "canvas":{"ratios":crate::h3_protocol::geometry::RATIOS,"qualities":crate::h3_protocol::geometry::QUALITIES,"default":imagine_video_contract::generation::Canvas::default()}}),
    )
}
/// Project-only admission blockers for a scene; see `execution::preflight`.
async fn preflight(
    State(app): State<App>,
    Path((id, scene)): Path<(String, String)>,
) -> Result<Json<Value>, Error> {
    let service = app.video.lock().await;
    let project = service.projects.read(&ProjectId(id))?;
    let scene = SceneId(scene);
    let selected = project.scene(&scene)?;
    let missing = service
        .h3_model_root
        .as_ref()
        .map(|root| super::model_files::missing(&project, selected, root))
        .unwrap_or_default();
    let mut blockers = super::execution::preflight(&project, &scene);
    if !missing.is_empty() {
        blockers.push("h3_model_file_missing".into());
    }
    Ok(Json(json!({
        "schema_version": 1,
        "project_revision": project.project_revision,
        "scene_id": scene,
        "blockers": blockers,
        "missing_model_files": missing,
    })))
}
async fn create(
    State(app): State<App>,
    body: Result<Json<CreateProject>, JsonRejection>,
) -> Result<(StatusCode, Json<Snapshot>), Error> {
    let request = input(body)?;
    let mut service = app.video.lock().await;
    let project = if let Some(root) = app.store.private_root() {
        service.projects.create_private(request, root)?
    } else {
        service.projects.create(request)?
    };
    Ok((StatusCode::CREATED, Json(service.snapshot(&project.id)?)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateAt {
    schema_version: u32,
    location_id: String,
    #[serde(default)]
    project: Option<CreateProject>,
    /// The default project on this canvas; exclusive with `project`.
    #[serde(default)]
    canvas: Option<imagine_video_contract::generation::Canvas>,
}
/// Create a project saved at once in an explicitly selected, new or empty folder.
async fn create_at(
    State(app): State<App>,
    body: Result<Json<CreateAt>, JsonRejection>,
) -> Result<(StatusCode, Json<Snapshot>), Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let create = match (request.project, request.canvas) {
        (Some(_), Some(_)) => {
            return Err(Error::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
            ));
        }
        (Some(project), None) => project,
        (None, Some(canvas)) => crate::video_defaults::with_canvas(canvas)
            .map_err(|code| Error::new(StatusCode::UNPROCESSABLE_ENTITY, code))?,
        (None, None) => crate::video_defaults::new_project(),
    };
    let mut service = app.video.lock().await;
    let path = service.location(&request.location_id)?;
    // A retried create finds the project this session already bound there.
    if let Some(id) = service.projects.at_location(&path)? {
        return Ok((StatusCode::OK, Json(service.snapshot(&id)?)));
    }
    service.projects.outside_projects(&path)?;
    let project = if let Some(root) = app.store.private_root() {
        let project = service.projects.create_private_at(create, root, &path)?;
        if let Some(location) = service.projects.session(&project.id)?.location() {
            app.store.bind_project_location(&project.id.0, location);
        }
        project
    } else {
        service.projects.create_at(create, &path)?
    };
    Ok((StatusCode::CREATED, Json(service.snapshot(&project.id)?)))
}
async fn read(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Snapshot>, Error> {
    Ok(Json(app.video.lock().await.snapshot(&ProjectId(id))?))
}
async fn compiled(
    State(app): State<App>,
    Path((id, scene)): Path<(String, String)>,
) -> Result<Json<super::adapters::h3::bindings::Plan>, Error> {
    let service = app.video.lock().await;
    let project = service.projects.read(&ProjectId(id))?;
    let scene = project.scene(&SceneId(scene))?;
    Ok(Json(super::adapters::h3::bindings::compile(
        &project, scene,
    )?))
}
async fn edit(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<EditRequest>, JsonRejection>,
) -> Result<Json<Snapshot>, Error> {
    let request = input(body)?;
    let id = ProjectId(id);
    let mut service = app.video.lock().await;
    if let Err(error) = service.projects.session_mut(&id)?.edit(request) {
        let mut error = Error::from(error);
        error.snapshot = service.snapshot(&id).ok().map(Box::new);
        return Err(error);
    }
    Ok(Json(service.snapshot(&id)?))
}
async fn review(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<super::review::ReviewRequest>, JsonRejection>,
) -> Result<Json<Snapshot>, Error> {
    let request = input(body)?;
    let id = ProjectId(id);
    let mut service = app.video.lock().await;
    if let Err(error) = service.projects.session_mut(&id)?.review(request) {
        let mut error = Error::from(error);
        error.snapshot = service.snapshot(&id).ok().map(Box::new);
        return Err(error);
    }
    Ok(Json(service.snapshot(&id)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectLocation {
    schema_version: u32,
    path: PathBuf,
    /// Create the missing leaf folder; its parent must already exist.
    #[serde(default)]
    create: bool,
}
async fn location(
    State(app): State<App>,
    body: Result<Json<SelectLocation>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let session_data = app
        .store
        .media
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| internal("session root"))?;
    let mut service = app.video.lock().await;
    if request.create {
        service.create_location(&request.path, session_data, app.store.private_root())?;
    }
    let id = service.select_location(&request.path, session_data)?;
    Ok((StatusCode::CREATED, Json(json!({"location_id":id}))))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Save {
    schema_version: u32,
    expected_project_revision: u64,
    location_id: Option<String>,
}
async fn save(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Save>, JsonRejection>,
) -> Result<Json<Snapshot>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let id = ProjectId(id);
    let mut service = app.video.lock().await;
    if service.projects.read(&id)?.project_revision != request.expected_project_revision {
        return Err(Error::new(StatusCode::CONFLICT, "revision_conflict"));
    }
    let path = request
        .location_id
        .as_deref()
        .map(|id| service.location(id))
        .transpose()?;
    let session = service.projects.session_mut(&id)?;
    let saved = if let Some(root) = app.store.private_root() {
        if let Some(path) = path
            && !session.is_at(&path)?
        {
            return Err(media_error("private_storage_location_required"));
        }
        if !session.is_bound() {
            session.save_private(root)
        } else {
            session.save()
        }
    } else if let Some(path) = path {
        session.save_to(&path)
    } else {
        session.save()
    };
    if let Err(error) = saved {
        let mut error = Error::from(error);
        error.snapshot = service.snapshot(&id).ok().map(Box::new);
        return Err(error);
    }
    Ok(Json(service.snapshot(&id)?))
}
async fn save_as(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Save>, JsonRejection>,
) -> Result<(StatusCode, Json<Snapshot>), Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let id = ProjectId(id);
    let staging = staging(&app)?;
    let mut service = app.video.lock().await;
    if service.projects.read(&id)?.project_revision != request.expected_project_revision {
        return Err(Error::new(StatusCode::CONFLICT, "revision_conflict"));
    }
    let copied = if let Some(root) = app.store.private_root() {
        if request.location_id.is_some() {
            return Err(media_error("private_storage_location_required"));
        }
        service.copy_private(&id, root, &staging)?
    } else {
        let location = request
            .location_id
            .ok_or_else(|| media_error("location_required"))?;
        let path = service.location(&location)?;
        service.copy_to(&id, &path, &staging)?
    };
    let bindings = service.runtime_assets.get(&id).cloned().unwrap_or_default();
    service.runtime_assets.insert(copied.clone(), bindings);
    let takes = service.runtime_takes.get(&id).cloned().unwrap_or_default();
    service.runtime_takes.insert(copied.clone(), takes);
    Ok((StatusCode::CREATED, Json(service.snapshot(&copied)?)))
}
async fn export_compatible(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Save>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let id = ProjectId(id);
    let mut service = app.video.lock().await;
    if service.projects.read(&id)?.project_revision != request.expected_project_revision {
        return Err(Error::new(StatusCode::CONFLICT, "revision_conflict"));
    }
    let report = if let Some(root) = app.store.private_root() {
        if request.location_id.is_some() {
            return Err(media_error("private_storage_location_required"));
        }
        service
            .projects
            .session_mut(&id)?
            .export_compatible_private(root)?
    } else {
        let location = request
            .location_id
            .ok_or_else(|| media_error("location_required"))?;
        let path = service.location(&location)?;
        service
            .projects
            .session_mut(&id)?
            .export_compatible(&path)?
    };
    let mut snapshot = serde_json::to_value(service.snapshot(&id)?).map_err(internal)?;
    snapshot["compatibility_export"] = serde_json::to_value(report).map_err(internal)?;
    Ok(Json(snapshot))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Open {
    schema_version: u32,
    location_id: Option<String>,
    project_id: Option<ProjectId>,
}
enum ResourceBinding {
    Reference(AssetId, u64),
    Take(TakeId),
}
async fn open(
    State(app): State<App>,
    body: Result<Json<Open>, JsonRejection>,
) -> Result<Json<Snapshot>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let path = {
        let service = app.video.lock().await;
        let path = if let Some(id) = request.project_id {
            let root = app
                .store
                .private_root()
                .ok_or_else(|| media_error("location_required"))?;
            root.proof().map_err(media_error)?;
            root.layout().project(&id.0).map_err(media_error)?.root
        } else {
            service.location(
                request
                    .location_id
                    .as_deref()
                    .ok_or_else(|| media_error("location_required"))?,
            )?
        };
        if let Some(id) = service.projects.at_location(&path)? {
            // Reattach the live session; reopening its disk snapshot would discard state.
            return Ok(Json(service.snapshot(&id)?));
        }
        path
    };
    let private = app.store.private_root().cloned();
    let project = tokio::task::spawn_blocking(move || match private {
        Some(root) => ProjectSession::open_for_private(&root, &path),
        None => ProjectSession::open(&path),
    })
    .await
    .map_err(internal)??;
    let mut service = app.video.lock().await;
    register_project(&app, &mut service, project).await
}
async fn register_project(
    app: &App,
    service: &mut super::service::Service,
    mut project: ProjectSession,
) -> Result<Json<Snapshot>, Error> {
    if let Some(root) = app.store.private_root()
        && !project.is_bound()
    {
        project.save_private(root)?;
    }
    let id = project.document().project.id.clone();
    if service.projects.session(&id).is_ok() {
        return Err(Error::new(StatusCode::CONFLICT, "project_already_open"));
    }
    let mut sources = vec![];
    let mut identities = vec![];
    let mut origins = vec![];
    for resource in &project.document().resources {
        let binding = match &resource.kind {
            ResourceKind::Reference { asset_id, revision } => {
                Some(ResourceBinding::Reference(asset_id.clone(), *revision))
            }
            ResourceKind::TakeVideo { take_id } => Some(ResourceBinding::Take(take_id.clone())),
            _ => None,
        };
        if let Some(binding) = binding {
            let file = match project.resource_file(&resource.id) {
                Ok(file) => file,
                // The project metadata remains editable so a missing/changed
                // original can be repaired explicitly with Check Media/Relink.
                Err(_) if resource.source.is_some() => continue,
                Err(error) => return Err(error.into()),
            };
            let (file, info) = tokio::task::spawn_blocking(move || {
                media::detect_streamed(&file).map(|info| (file, info))
            })
            .await
            .map_err(internal)?
            .map_err(media_error)?;
            sources.push((file, info));
            identities.push(binding);
            origins.push(resource.source.clone());
        }
    }
    // Inspect every file before registering any asset. Cancellation or an invalid later
    // resource cannot leave a partially reopened document or a half-registered batch.
    let values = app
        .store
        .register_media_batch("local", sources)
        .await
        .map_err(internal)?;
    let mut bindings = BTreeMap::new();
    let mut takes = BTreeMap::new();
    for ((binding, value), origin) in identities.into_iter().zip(values).zip(origins) {
        if let Some(path) = origin {
            app.store
                .remember_original(value["id"].as_str().unwrap(), path, &value)
                .map_err(internal)?;
        }
        match binding {
            ResourceBinding::Reference(asset_id, revision)
                if project
                    .document()
                    .project
                    .assets
                    .get(&asset_id)
                    .is_some_and(|a| a.revision == revision) =>
            {
                bindings.insert(asset_id, value["id"].as_str().unwrap().to_owned());
            }
            ResourceBinding::Take(take_id) => {
                takes.insert(take_id, value["id"].as_str().unwrap().to_owned());
            }
            _ => {}
        };
    }
    let located = app
        .store
        .private_root()
        .and_then(|_| project.document().storage_layout)
        .and_then(|_| project.location());
    service.projects.insert(project)?;
    if let Some(location) = located {
        app.store.bind_project_location(&id.0, location);
    }
    service.runtime_assets.insert(id.clone(), bindings);
    service.runtime_takes.insert(id.clone(), takes);
    Ok(Json(service.snapshot(&id)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Attach {
    #[serde(default)]
    library_source: Option<LibraryMediaSource>,
    #[serde(default)]
    name: String,
    schema_version: u32,
    expected_project_revision: u64,
    asset_id: String,
}
async fn attach(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Attach>, JsonRejection>,
) -> Result<Json<Snapshot>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let id = ProjectId(id);
    let info = app
        .store
        .asset(&request.asset_id, "local")
        .await
        .map_err(internal)?
        .ok_or_else(|| Error::new(StatusCode::NOT_FOUND, "asset_not_found"))?;
    let staging = staging(&app)?;
    let mut file = if let Some(file) = app.store.open_media(&request.asset_id).map_err(internal)? {
        file
    } else {
        let bytes = app
            .store
            .upload_bytes(&request.asset_id)
            .map_err(internal)?
            .ok_or_else(|| media_error("asset_not_found"))?;
        let mut file = staging.temporary()?;
        file.write_all(&bytes).map_err(internal)?;
        file
    };
    let mime = info["mime_type"].as_str().unwrap_or("").to_owned();
    let (returned, inspected) = tokio::task::spawn_blocking(move || {
        media::inspect_streamed(&file, &mime).map(|info| (file, info))
    })
    .await
    .map_err(internal)?
    .map_err(media_error)?;
    file = returned;
    let asset_id = AssetId(format!("project_asset_{}", uuid::Uuid::new_v4().simple()));
    let resource_id = format!("resource_{}", uuid::Uuid::new_v4().simple());
    let resource_kind = ResourceKind::Reference {
        asset_id: asset_id.clone(),
        revision: 1,
    };
    let resource = if let Some(path) = app
        .store
        .original_path(&request.asset_id)
        .map_err(internal)?
    {
        Resource::external(resource_id, resource_kind, &path, &mut file)?
    } else {
        Resource::inspect(resource_id, resource_kind, &mut file)?
    };
    if resource.sha256 != info["sha256"].as_str().unwrap_or("")
        || Some(resource.bytes) != info["bytes"].as_u64()
    {
        return Err(media_error("asset_content_changed"));
    }
    let mut service = app.video.lock().await;
    let session = service.projects.session_mut(&id)?;
    let mut document = session.document().clone();
    document.project.assets.insert(
        asset_id.clone(),
        Asset {
            library_source: request.library_source,
            name: request.name,
            id: asset_id.clone(),
            revision: 1,
            kind: inspected.kind,
        },
    );
    let key = resource.key();
    document.resources.push(resource);
    let updated = session.update(
        request.expected_project_revision,
        document,
        BTreeMap::from([(key, file)]),
        &staging,
    );
    // A failed autosave can retain the applied draft, including its runtime binding.
    if service
        .projects
        .session(&id)?
        .document()
        .project
        .assets
        .contains_key(&asset_id)
    {
        service
            .runtime_assets
            .entry(id.clone())
            .or_default()
            .insert(asset_id, request.asset_id);
    }
    if let Err(error) = updated {
        let mut error = Error::from(error);
        error.snapshot = service.snapshot(&id).ok().map(Box::new);
        return Err(error);
    }
    Ok(Json(service.snapshot(&id)?))
}
async fn upload(
    State(app): State<App>,
    request: Request,
) -> Result<(StatusCode, Json<Value>), Error> {
    streaming_import::upload(app, request).await
}

async fn asset(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let mut value = app
        .store
        .asset(&id, "local")
        .await
        .map_err(internal)?
        .ok_or_else(|| Error::new(StatusCode::NOT_FOUND, "asset_not_found"))?;
    let file = app.store.open_media(&id).map_err(internal)?;
    let file = if let Some(file) = file {
        file
    } else {
        let bytes = app
            .store
            .upload_bytes(&id)
            .map_err(internal)?
            .ok_or_else(|| Error::new(StatusCode::NOT_FOUND, "asset_not_found"))?;
        let mut file = staging(&app)?.temporary()?;
        file.write_all(&bytes).map_err(internal)?;
        file
    };
    let mime = value["mime_type"].as_str().unwrap_or("").to_owned();
    let (info, digest, bytes) = tokio::task::spawn_blocking(move || {
        let mut file = file;
        let info = media::inspect_streamed(&file, &mime).map_err(media_error)?;
        let (digest, bytes) = super::bundle::digest(&mut file)?;
        Ok::<_, Error>((info, digest, bytes))
    })
    .await
    .map_err(internal)??;
    if value["sha256"].as_str() != Some(&digest) || value["bytes"].as_u64() != Some(bytes) {
        return Err(media_error("asset_content_changed"));
    }
    value["media"] = json!(info);
    Ok(Json(value))
}
