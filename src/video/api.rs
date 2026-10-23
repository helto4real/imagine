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
use tokio::io::AsyncWriteExt;
#[path = "assembly_api.rs"]
mod assembly;
#[path = "authoring_api.rs"]
mod authoring;
#[path = "library_api.rs"]
mod libraries;
#[path = "runs_api.rs"]
mod runs;

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
    let proof = app
        .store
        .videos
        .destination()
        .and_then(|d| d.proof())
        .map_err(media_error)?;
    SessionStaging::select(&proof.config.root).map_err(Into::into)
}
pub fn routes() -> Router<App> {
    Router::new()
        .merge(libraries::routes())
        .merge(runs::routes())
        .merge(authoring::routes())
        .merge(assembly::routes())
        .route("/v1/video/capabilities", get(capabilities))
        .route("/v1/video/locations", post(location))
        .route("/v1/video/projects", post(create))
        .route("/v1/video/projects/open", post(open))
        .route("/v1/video/projects/{id}", get(read))
        .route("/v1/video/projects/{id}/edits", post(edit))
        .route(
            "/v1/video/projects/{id}/scenes/{scene}/compiled",
            get(compiled),
        )
        .route("/v1/video/projects/{id}/review", post(review))
        .route("/v1/video/projects/{id}/assets", post(attach))
        .route("/v1/video/projects/{id}/save", post(save))
        .route("/v1/video/projects/{id}/save-as", post(save_as))
        .route("/v1/video/assets", post(upload))
        .route("/v1/video/assets/{id}", get(asset))
}
async fn capabilities() -> Json<Value> {
    Json(
        json!({"schema_version":1,"media":{"mime_types":media::MIME_TYPES,"max_upload_bytes":media::MAX_UPLOAD},
        "project_operations":["create","read","edits","review","attach_asset","save","open","save_as"],"models":[]}),
    )
}
async fn create(
    State(app): State<App>,
    body: Result<Json<CreateProject>, JsonRejection>,
) -> Result<(StatusCode, Json<Snapshot>), Error> {
    let request = input(body)?;
    let mut service = app.video.lock().await;
    let project = service.projects.create(request)?;
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
    let id = app
        .video
        .lock()
        .await
        .select_location(&request.path, session_data)?;
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
    let saved = if let Some(path) = path {
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
    let location = request
        .location_id
        .ok_or_else(|| media_error("location_required"))?;
    let path = service.location(&location)?;
    let copied = service.copy_to(&id, &path, &staging)?;
    let bindings = service.runtime_assets.get(&id).cloned().unwrap_or_default();
    service.runtime_assets.insert(copied.clone(), bindings);
    let takes = service.runtime_takes.get(&id).cloned().unwrap_or_default();
    service.runtime_takes.insert(copied.clone(), takes);
    Ok((StatusCode::CREATED, Json(service.snapshot(&copied)?)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Open {
    schema_version: u32,
    location_id: String,
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
        let path = service.location(&request.location_id)?;
        if let Some(id) = service.projects.at_location(&path)? {
            // Reattach the live session; reopening its disk snapshot would discard state.
            return Ok(Json(service.snapshot(&id)?));
        }
        path
    };
    let project = tokio::task::spawn_blocking(move || ProjectSession::open(&path))
        .await
        .map_err(internal)??;
    let mut service = app.video.lock().await;
    register_project(&app, &mut service, project).await
}
async fn register_project(
    app: &App,
    service: &mut super::service::Service,
    project: ProjectSession,
) -> Result<Json<Snapshot>, Error> {
    let id = project.document().project.id.clone();
    if service.projects.session(&id).is_ok() {
        return Err(Error::new(StatusCode::CONFLICT, "project_already_open"));
    }
    let mut sources = vec![];
    let mut identities = vec![];
    for resource in &project.document().resources {
        let binding = match &resource.kind {
            ResourceKind::Reference { asset_id, revision } => {
                Some(ResourceBinding::Reference(asset_id.clone(), *revision))
            }
            ResourceKind::TakeVideo { take_id } => Some(ResourceBinding::Take(take_id.clone())),
            _ => None,
        };
        if let Some(binding) = binding {
            let file = project.resource_file(&resource.id)?;
            let (file, info) =
                tokio::task::spawn_blocking(move || media::detect(&file).map(|info| (file, info)))
                    .await
                    .map_err(internal)?
                    .map_err(media_error)?;
            sources.push((file, info));
            identities.push(binding);
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
    for (binding, value) in identities.into_iter().zip(values) {
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
    service.projects.insert(project)?;
    service.runtime_assets.insert(id.clone(), bindings);
    service.runtime_takes.insert(id.clone(), takes);
    Ok(Json(service.snapshot(&id)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Attach {
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
    let (returned, inspected) =
        tokio::task::spawn_blocking(move || media::inspect(&file, &mime).map(|info| (file, info)))
            .await
            .map_err(internal)?
            .map_err(media_error)?;
    file = returned;
    let asset_id = AssetId(format!("project_asset_{}", uuid::Uuid::new_v4().simple()));
    let resource = Resource::inspect(
        format!("resource_{}", uuid::Uuid::new_v4().simple()),
        ResourceKind::Reference {
            asset_id: asset_id.clone(),
            revision: 1,
        },
        &mut file,
    )?;
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
    let mime = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    if !media::MIME_TYPES.contains(&mime.as_str()) {
        return Err(Error::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
        ));
    }
    if let Some(length) = request.headers().get(header::CONTENT_LENGTH) {
        let length = length
            .to_str()
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| Error::new(StatusCode::BAD_REQUEST, "invalid_content_length"))?;
        if length > media::MAX_UPLOAD {
            return Err(Error::new(StatusCode::PAYLOAD_TOO_LARGE, "media_too_large"));
        }
    }
    let _permit = app
        .uploads
        .clone()
        .try_acquire_owned()
        .map_err(|_| Error::new(StatusCode::TOO_MANY_REQUESTS, "upload_busy"))?;
    let staging = staging(&app)?;
    let mut file = tokio::fs::File::from_std(staging.temporary()?);
    let mut stream = request.into_body().into_data_stream();
    let mut bytes = 0u64;
    while let Some(chunk) = tokio::time::timeout(std::time::Duration::from_secs(30), stream.next())
        .await
        .map_err(|_| media_error("upload_timeout"))?
    {
        let chunk = chunk.map_err(|_| media_error("upload_interrupted"))?;
        bytes = bytes
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| media_error("media_too_large"))?;
        if bytes > media::MAX_UPLOAD {
            return Err(Error::new(StatusCode::PAYLOAD_TOO_LARGE, "media_too_large"));
        }
        file.write_all(&chunk).await.map_err(internal)?;
    }
    file.flush().await.map_err(internal)?;
    let file = file.into_std().await;
    let (file, info) =
        tokio::task::spawn_blocking(move || media::inspect(&file, &mime).map(|info| (file, info)))
            .await
            .map_err(internal)?
            .map_err(media_error)?;
    let result = app
        .store
        .register_media("local", file, &info)
        .await
        .map_err(internal)?;
    Ok((StatusCode::CREATED, Json(result)))
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
        let info = media::inspect(&file, &mime).map_err(media_error)?;
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
