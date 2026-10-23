//! Explicit local library selections. No filesystem discovery or live project dependencies.
use super::{Error, input, internal, media_error, register_project, schema, staging};
use crate::{
    api::App,
    video::{contract::*, library::*, media, persistence::*},
};
use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::StatusCode,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/libraries", post(select))
        .route("/v1/video/libraries/{id}", get(read))
        .route("/v1/video/libraries/{id}/projects", post(save_project))
        .route("/v1/video/libraries/{id}/characters", post(save_character))
        .route(
            "/v1/video/projects/{id}/cast/from-library",
            post(attach_character),
        )
        .route(
            "/v1/video/libraries/{id}/entries/{entry}/project-copy",
            post(copy_project),
        )
        .route(
            "/v1/video/libraries/{id}/entries/{entry}/character-copy",
            post(copy_character),
        )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AttachCharacter {
    schema_version: u32,
    expected_project_revision: u64,
    library_id: String,
    entry_id: String,
    expected_entry_revision: u64,
}
async fn attach_character(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<AttachCharacter>, JsonRejection>,
) -> Result<Json<crate::video::service::Snapshot>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let staging = staging(&app)?;
    let mut service = app.video.lock().await;
    let id = ProjectId(id);
    let mut prepared = super::super::cast::from_library(
        &mut service,
        &request.library_id,
        &request.entry_id,
        request.expected_entry_revision,
        &id,
        request.expected_project_revision,
        &staging,
    )?;
    let values = app
        .store
        .register_media_batch("local", std::mem::take(&mut prepared.files))
        .await
        .map_err(internal)?;
    if let Err(error) = prepared.commit(&mut service, values) {
        let mut error = Error::from(error);
        error.snapshot = Some(Box::new(service.snapshot(&id)?));
        return Err(error);
    }
    Ok(Json(service.snapshot(&id)?))
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Create,
    Open,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    schema_version: u32,
    location_id: String,
    mode: Mode,
}
async fn select(
    State(app): State<App>,
    body: Result<Json<Selection>, JsonRejection>,
) -> Result<Json<LibraryDocument>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let path = app.video.lock().await.location(&request.location_id)?;
    let library = tokio::task::spawn_blocking(move || match request.mode {
        Mode::Create => LibrarySession::create(&path),
        Mode::Open => LibrarySession::open(&path),
    })
    .await
    .map_err(internal)??;
    let document = library.document().clone();
    let mut service = app.video.lock().await;
    if service.libraries.contains_key(&document.id) {
        return Err(Error::new(StatusCode::CONFLICT, "library_already_open"));
    }
    service.libraries.insert(document.id.clone(), library);
    Ok(Json(document))
}
async fn read(
    State(app): State<App>,
    Path(id): Path<String>,
) -> Result<Json<LibraryDocument>, Error> {
    let service = app.video.lock().await;
    Ok(Json(
        service
            .libraries
            .get(&id)
            .ok_or_else(missing)?
            .document()
            .clone(),
    ))
}
fn missing() -> Error {
    Error::new(StatusCode::NOT_FOUND, "library_not_found")
}
fn revision(actual: u64, expected: u64) -> Result<(), Error> {
    if actual != expected {
        Err(Error::new(StatusCode::CONFLICT, "revision_conflict"))
    } else {
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Replace {
    entry_id: String,
    expected_entry_revision: u64,
}
impl Replace {
    fn pair(&self) -> (&str, u64) {
        (&self.entry_id, self.expected_entry_revision)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveProject {
    schema_version: u32,
    expected_library_revision: u64,
    project_id: ProjectId,
    expected_project_revision: u64,
    title: String,
    replace: Option<Replace>,
}
async fn save_project(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<SaveProject>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let mut service = app.video.lock().await;
    let crate::video::service::Service {
        projects,
        libraries,
        ..
    } = &mut *service;
    let project = projects.session(&request.project_id)?;
    revision(
        project.document().project.project_revision,
        request.expected_project_revision,
    )?;
    let library = libraries.get_mut(&id).ok_or_else(missing)?;
    revision(
        library.document().revision,
        request.expected_library_revision,
    )?;
    let entry = library.save_project(
        request.title,
        request.replace.as_ref().map(Replace::pair),
        project,
    )?;
    Ok(Json(json!({"entry_id":entry,"library":library.document()})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CopyEntry {
    schema_version: u32,
    expected_entry_revision: u64,
}
async fn copy_project(
    State(app): State<App>,
    Path((id, entry)): Path<(String, String)>,
    body: Result<Json<CopyEntry>, JsonRejection>,
) -> Result<Json<crate::video::service::Snapshot>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let staging = staging(&app)?;
    let mut service = app.video.lock().await;
    let project = service
        .libraries
        .get(&id)
        .ok_or_else(missing)?
        .copy_project(&entry, request.expected_entry_revision, &staging)?;
    register_project(&app, &mut service, project).await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Voice {
    language: String,
    description: String,
    reference_asset_ids: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Character {
    name: String,
    alias: String,
    description: String,
    tags: Vec<String>,
    visual_asset_ids: Vec<String>,
    voice: Option<Voice>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveCharacter {
    schema_version: u32,
    expected_library_revision: u64,
    title: String,
    replace: Option<Replace>,
    character: Character,
}
async fn save_character(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<SaveCharacter>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let staging = staging(&app)?;
    let mut inputs = BTreeMap::new();
    let mut resources = vec![];
    let mut visuals = vec![];
    let mut voices = vec![];
    let visual = &request.character.visual_asset_ids;
    let voice = request
        .character
        .voice
        .as_ref()
        .map(|v| v.reference_asset_ids.as_slice())
        .unwrap_or_default();
    for (ids, kind) in [
        (visual.as_slice(), MediaKind::Image),
        (voice, MediaKind::Audio),
    ] {
        for id in ids {
            let info = app
                .store
                .asset(id, "local")
                .await
                .map_err(internal)?
                .ok_or_else(|| media_error("asset_not_found"))?;
            let mut file = if let Some(file) = app.store.open_media(id).map_err(internal)? {
                file
            } else {
                use std::io::Write;
                let bytes = app
                    .store
                    .upload_bytes(id)
                    .map_err(internal)?
                    .ok_or_else(|| media_error("asset_not_found"))?;
                let mut file = staging.temporary()?;
                file.write_all(&bytes).map_err(internal)?;
                file
            };
            let mime = info["mime_type"].as_str().unwrap_or("").to_owned();
            let (returned, typed) = tokio::task::spawn_blocking(move || {
                media::inspect(&file, &mime).map(|typed| (file, typed))
            })
            .await
            .map_err(internal)?
            .map_err(media_error)?;
            file = returned;
            if typed.kind != kind {
                return Err(media_error("character_media_type_mismatch"));
            }
            let resource = Resource::inspect(
                format!("reference_{}", uuid::Uuid::new_v4().simple()),
                ResourceKind::Reference {
                    asset_id: AssetId(format!("character_asset_{}", uuid::Uuid::new_v4().simple())),
                    revision: 1,
                },
                &mut file,
            )?;
            if Some(resource.sha256.as_str()) != info["sha256"].as_str()
                || Some(resource.bytes) != info["bytes"].as_u64()
            {
                return Err(media_error("asset_content_changed"));
            }
            if kind == MediaKind::Image {
                visuals.push(resource.id.clone());
            } else {
                voices.push(resource.id.clone());
            }
            inputs.insert(resource.key(), file);
            resources.push(resource);
        }
    }
    let character = request.character;
    let mut document = CharacterDocument {
        schema_version: 1,
        id: format!("character_{}", uuid::Uuid::new_v4().simple()),
        name: character.name,
        alias: character.alias,
        description: character.description,
        tags: character.tags,
        visual_ids: visuals,
        voice: character.voice.map(|v| VoiceProfile {
            language: v.language,
            description: v.description,
            reference_ids: voices,
        }),
        resources,
    };
    let mut service = app.video.lock().await;
    let library = service.libraries.get_mut(&id).ok_or_else(missing)?;
    revision(
        library.document().revision,
        request.expected_library_revision,
    )?;
    if let Some(replace) = &request.replace
        && let Some(entry) = library
            .document()
            .entries
            .iter()
            .find(|e| e.id == replace.entry_id)
        && let LibraryContent::Character(character) = &entry.content
    {
        document.id = character.id.clone();
    }
    let entry = library.save_character(
        request.title,
        request.replace.as_ref().map(Replace::pair),
        document,
        inputs,
    )?;
    Ok(Json(json!({"entry_id":entry,"library":library.document()})))
}
async fn copy_character(
    State(app): State<App>,
    Path((id, entry)): Path<(String, String)>,
    body: Result<Json<CopyEntry>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let staging = staging(&app)?;
    let service = app.video.lock().await;
    let copy = service
        .libraries
        .get(&id)
        .ok_or_else(missing)?
        .copy_character(&entry, request.expected_entry_revision, &staging)?;
    let mut files = vec![];
    let mut ids = vec![];
    for resource in &copy.character.resources {
        let source = copy
            .sources
            .get(&resource.key())
            .ok_or_else(|| media_error("resource_missing"))?
            .try_clone()
            .map_err(internal)?;
        let (file, info) =
            tokio::task::spawn_blocking(move || media::detect(&source).map(|info| (source, info)))
                .await
                .map_err(internal)?
                .map_err(media_error)?;
        files.push((file, info));
        ids.push(resource.id.clone());
    }
    let values = app
        .store
        .register_media_batch("local", files)
        .await
        .map_err(internal)?;
    let assets: BTreeMap<_, _> = ids.into_iter().zip(values).collect();
    Ok(Json(
        json!({"character":copy.character,"origin":copy.origin,"runtime_assets":assets}),
    ))
}
