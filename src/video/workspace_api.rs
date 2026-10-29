//! Agent workspace routes. Keyed by workspace ID; the MCP proxy resolves an
//! agent's key to that ID and never passes paths back to the agent. The UI
//! uses the listing and key routes.
use super::*;
use crate::workspaces::{Workspace, Workspaces};
use axum::{
    body::Body,
    extract::Query,
    http::{HeaderMap, header},
};
use futures_util::StreamExt;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;

pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/workspaces", post(create).get(list))
        .route("/v1/workspaces/resolve", post(resolve))
        .route("/v1/workspaces/{id}/key", get(key).post(replace_key))
        .route("/v1/workspaces/{id}/files", get(files).put(upload))
        .route("/v1/workspaces/{id}/files/{file}", get(download))
        .route("/v1/workspaces/{id}/files/{file}/info", get(info))
        .route("/v1/workspaces/{id}/destination", get(destination))
        .route("/v1/workspaces/{id}/video-projects", post(video_project))
}
fn store(app: &App) -> Result<&Workspaces, Error> {
    app.store
        .workspaces
        .as_deref()
        .ok_or_else(|| Error::new(StatusCode::SERVICE_UNAVAILABLE, "mcp_root_unavailable"))
}
fn failure(error: crate::workspaces::Error) -> Error {
    let status = match error.0 {
        "workspace_unknown" | "workspace_not_found" | "file_not_found" => StatusCode::NOT_FOUND,
        "upload_too_large" => StatusCode::PAYLOAD_TOO_LARGE,
        "upload_type_unsupported" => StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "private_storage_exists" => StatusCode::CONFLICT,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    };
    Error::new(status, error.0)
}
fn workspace(app: &App, id: &str) -> Result<Workspace, Error> {
    store(app)?.get(id).map_err(failure)
}
/// The proxy-facing description: includes the folder, which the proxy keeps
/// to itself.
fn described(workspace: &Workspace) -> Value {
    workspace.summary()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Create {
    schema_version: u32,
    #[serde(default)]
    title: String,
}
async fn create(
    State(app): State<App>,
    body: Result<Json<Create>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let (workspace, key) = store(&app)?.create(&request.title).map_err(failure)?;
    let mut value = described(&workspace);
    value["key"] = json!(key);
    Ok((StatusCode::CREATED, Json(value)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Resolve {
    schema_version: u32,
    key: String,
}
async fn resolve(
    State(app): State<App>,
    body: Result<Json<Resolve>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let workspace = store(&app)?.resolve(&request.key).map_err(failure)?;
    Ok(Json(described(&workspace)))
}
async fn list(State(app): State<App>) -> Result<Json<Value>, Error> {
    let workspaces = store(&app)?.list();
    Ok(Json(
        json!({"schema_version":1,"workspaces":workspaces.iter().map(Workspace::summary).collect::<Vec<_>>()}),
    ))
}
async fn key(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let key = store(&app)?.key(&id).map_err(failure)?;
    Ok(Json(json!({"schema_version":1,"key":key})))
}
async fn replace_key(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let key = store(&app)?.replace_key(&id).map_err(failure)?;
    Ok(Json(json!({"schema_version":1,"key":key})))
}
async fn files(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let workspace = workspace(&app, &id)?;
    let files = tokio::task::spawn_blocking({
        let app = app.clone();
        move || store(&app).map(|w| w.files(&workspace))
    })
    .await
    .map_err(internal)??;
    Ok(Json(json!({"schema_version":1,"files":files})))
}
async fn info(
    State(app): State<App>,
    Path((id, file)): Path<(String, String)>,
) -> Result<Json<Value>, Error> {
    let workspace = workspace(&app, &id)?;
    let entry = store(&app)?.file(&workspace, &file).map_err(failure)?;
    let mut value = json!(entry);
    value["path"] = json!(entry.path);
    Ok(Json(value))
}
async fn download(
    State(app): State<App>,
    Path((id, file)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, Error> {
    let workspace = workspace(&app, &id)?;
    let entry = store(&app)?.file(&workspace, &file).map_err(failure)?;
    let opened = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&entry.path)
        .map_err(|_| Error::new(StatusCode::NOT_FOUND, "file_not_found"))?;
    let mut response = crate::video::http::ranged(
        Some(opened),
        None,
        entry.bytes,
        entry.mime_type,
        &format!("\"{}\"", entry.sha256),
        &headers,
    )
    .await
    .map_err(|fault| {
        Error::new(
            StatusCode::from_u16(fault.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            fault.code,
        )
    })?;
    if let Ok(value) = format!("attachment; filename=\"{}\"", entry.name).parse() {
        response
            .headers_mut()
            .insert(header::CONTENT_DISPOSITION, value);
    }
    Ok(response)
}
#[derive(Deserialize)]
struct UploadQuery {
    #[serde(default)]
    name: String,
}
/// Streams an upload into `uploads/`. The type comes from the content; the
/// size limit from that type.
async fn upload(
    State(app): State<App>,
    Path(id): Path<String>,
    Query(query): Query<UploadQuery>,
    body: Body,
) -> Result<(StatusCode, Json<Value>), Error> {
    let workspace = workspace(&app, &id)?;
    let mut stream = body.into_data_stream();
    let mut head = Vec::new();
    while head.len() < 16 {
        match stream.next().await {
            Some(Ok(chunk)) => head.extend_from_slice(&chunk),
            Some(Err(_)) => return Err(Error::new(StatusCode::BAD_REQUEST, "upload_interrupted")),
            None => break,
        }
    }
    let workspaces = store(&app)?;
    let (mut file, name, limit) = workspaces
        .upload(&workspace, &query.name, &head)
        .map_err(failure)?;
    // `None` ends a complete upload. The sender disappearing without it (a failed
    // body, a dropped request) leaves the file uncommitted, and dropping it
    // removes the temporary file.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Option<axum::body::Bytes>>(8);
    let writer = tokio::task::spawn_blocking(move || {
        let mut written = 0u64;
        loop {
            match rx.blocking_recv() {
                Some(Some(chunk)) => {
                    written += chunk.len() as u64;
                    if written > limit {
                        return Err("upload_too_large");
                    }
                    file.file()
                        .write_all(&chunk)
                        .map_err(|_| "private_storage_write_failed")?;
                }
                Some(None) => break,
                None => return Err("upload_interrupted"),
            }
        }
        if written == 0 {
            return Err("upload_empty");
        }
        file.commit()?;
        Ok(written)
    });
    let mut interrupted = false;
    if tx.send(Some(head.into())).await.is_ok() {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(chunk) => {
                    if tx.send(Some(chunk)).await.is_err() {
                        break;
                    }
                }
                Err(_) => {
                    interrupted = true;
                    break;
                }
            }
        }
    }
    if interrupted {
        drop(tx);
        let _ = writer.await;
        return Err(Error::new(StatusCode::BAD_REQUEST, "upload_interrupted"));
    }
    // A writer that already stopped (too large, write failure) reports why.
    let _ = tx.send(None).await;
    drop(tx);
    writer
        .await
        .map_err(internal)?
        .map_err(|code| failure(crate::workspaces::Error(code)))?;
    let path = workspace.folder.join("uploads").join(&name);
    let entry = workspaces
        .file_at(&workspace, &path)
        .ok_or_else(|| Error::new(StatusCode::INTERNAL_SERVER_ERROR, "storage_error"))?;
    Ok((StatusCode::CREATED, Json(json!(entry))))
}
/// The media destination proof for the workspace's image folder, which image
/// jobs pass as their output destination.
async fn destination(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let workspace = workspace(&app, &id)?;
    let root = app
        .store
        .private_root()
        .ok_or_else(|| media_error("private_storage_required"))?;
    let proof = crate::video::output::Destination::private(root, &workspace.images())
        .and_then(|d| d.proof())
        .map_err(media_error)?;
    Ok(Json(json!(proof)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VideoProject {
    schema_version: u32,
    #[serde(default)]
    title: String,
    #[serde(default)]
    project: Option<CreateProject>,
    #[serde(default)]
    canvas: Option<imagine_video_contract::generation::Canvas>,
}
/// A new video project in `videos/<title>` of the workspace.
async fn video_project(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<VideoProject>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Error> {
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
    let workspace = workspace(&app, &id)?;
    // A create retried after a lost reply finds the project it made, still
    // open in this session under the same title.
    let named = workspace
        .videos()
        .join(crate::workspaces::slug(&request.title, 40));
    let retried = !request.title.trim().is_empty()
        && app
            .video
            .lock()
            .await
            .projects
            .at_location(&named)?
            .is_some();
    let folder = if retried {
        named
    } else {
        store(&app)?
            .project_folder(&workspace, &request.title)
            .map_err(failure)?
    };
    let (status, Json(snapshot)) = create_at_path(&app, &folder, create).await?;
    let mut value = json!(snapshot);
    value["path"] = json!(folder);
    value["folder"] = json!(folder.file_name().and_then(|n| n.to_str()));
    Ok((status, Json(value)))
}
