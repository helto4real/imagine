//! Agent workspaces: every file an agent sends or receives lives in one
//! workspace, reached only with its key. Agents never see host paths; files
//! are named by ID and move through these tools or the HTTP file routes.
use super::*;
use axum::extract::Path as UrlPath;
use axum::response::IntoResponse;
use reqwest::Method;
use std::path::Path;

/// The encrypted volume; no path below it is ever shown to an agent.
const PRIVATE_MOUNT: &str = "/run/media/veracrypt1";
/// `workspace_upload` carries base64 in the tool call; larger files use HTTP.
const MAX_TOOL_UPLOAD: usize = 4 << 20;
const PATH_KEYS: &[&str] = &[
    "path",
    "saved_path",
    "preview_path",
    "source_path",
    "persistent_root",
    "location",
    "folder",
];

#[derive(Clone, Debug)]
pub(crate) struct Bound {
    pub(crate) id: String,
    pub(crate) folder: PathBuf,
}
fn key_of(args: &Value) -> std::result::Result<&str, Fault> {
    args["workspace_key"]
        .as_str()
        .filter(|key| crate::workspace_ids::key_shaped(key))
        .ok_or_else(|| {
            validation(
                "workspace_key is required: the imk-… key from workspace_create. Ask the user if you lost it.",
            )
        })
}
fn file_ident(args: &Value) -> std::result::Result<&str, Fault> {
    args["file_id"]
        .as_str()
        .filter(|id| {
            id.len() == 21 && id.starts_with('f') && id.bytes().all(|b| b.is_ascii_alphanumeric())
        })
        .ok_or_else(|| validation("file_id must be a file ID from this workspace"))
}
fn unknown() -> Fault {
    Fault::new(
        "not_found",
        "unknown workspace_key; use the exact key you saved, or ask the user for it",
    )
}
impl Mcp {
    pub(super) fn download_url(&self, file_id: &str) -> String {
        format!(
            "http://127.0.0.1:{}/v1/workspace/files/{file_id}",
            self.inner.config.mcp_port.unwrap_or(0)
        )
    }
    /// The agent's view of a server file entry: never its path.
    pub(super) fn file_view(&self, entry: &Value) -> Value {
        let id = entry["file_id"].as_str().unwrap_or_default();
        json!({"file_id":id,"name":entry["name"],"kind":entry["kind"],"origin":entry["origin"],
            "project":entry["project"],"mime_type":entry["mime_type"],"bytes":entry["bytes"],
            "sha256":entry["sha256"],"created_ms":entry["created_ms"],"download_url":self.download_url(id)})
    }
    /// The workspace for the call's `workspace_key`.
    pub(super) async fn bound(&self, args: &Value) -> std::result::Result<Bound, Fault> {
        self.resolve_key(key_of(args)?).await
    }
    pub(super) async fn resolve_key(&self, key: &str) -> std::result::Result<Bound, Fault> {
        let value = self
            .client
            .json(
                Method::POST,
                "/v1/workspaces/resolve",
                Some(&json!({"schema_version":1,"key":key})),
                None,
            )
            .await
            .map_err(|error| match crate::imagine::rejection(&error) {
                Some(r) if r.status == 404 => unknown(),
                Some(r) => Fault::new(
                    "imagine",
                    format!("Imagine rejected the request: {}", r.code),
                ),
                None => Fault::new("connection", "Imagine did not respond; retry"),
            })?;
        Ok(Bound {
            id: value["workspace_id"]
                .as_str()
                .ok_or_else(|| Fault::new("imagine", "invalid workspace response"))?
                .to_owned(),
            folder: value["path"]
                .as_str()
                .map(PathBuf::from)
                .ok_or_else(|| Fault::new("imagine", "invalid workspace response"))?,
        })
    }
    /// The real path of a workspace file, for Imagine-side registration only.
    pub(super) async fn file_path(
        &self,
        workspace: &Bound,
        file_id: &str,
    ) -> std::result::Result<(PathBuf, Value), Fault> {
        let entry = self
            .client
            .json(
                Method::GET,
                &format!("/v1/workspaces/{}/files/{file_id}/info", workspace.id),
                None,
                None,
            )
            .await
            .map_err(|error| match crate::imagine::rejection(&error) {
                Some(r) if r.status == 404 => {
                    Fault::new("not_found", "file_id is not a file of this workspace")
                }
                _ => Fault::new("connection", "Imagine did not respond; retry"),
            })?;
        let path = entry["path"]
            .as_str()
            .map(PathBuf::from)
            .filter(|p| p.starts_with(&workspace.folder))
            .ok_or_else(|| Fault::new("imagine", "invalid workspace file response"))?;
        Ok((path, entry))
    }
    /// The project snapshot, if `project_id` belongs to this workspace. A
    /// project saved earlier but not open in this Imagine session is reopened.
    pub(super) async fn owned_project(
        &self,
        workspace: &Bound,
        project_id: &str,
    ) -> std::result::Result<Value, Fault> {
        let path = format!("/v1/video/projects/{project_id}");
        let snapshot = match self.client.json(Method::GET, &path, None, None).await {
            Ok(snapshot) => snapshot,
            Err(error) if crate::imagine::rejection(&error).is_some_and(|r| r.status == 404) => {
                self.reopen(workspace, project_id).await?
            }
            Err(_) => return Err(Fault::new("connection", "Imagine did not respond; retry")),
        };
        let inside = snapshot["persistence"]["location"]
            .as_str()
            .is_some_and(|location| Path::new(location).starts_with(&workspace.folder));
        if !inside {
            return Err(Fault::new(
                "not_found",
                "project_id is not a project of this workspace",
            ));
        }
        Ok(snapshot)
    }
    async fn reopen(
        &self,
        workspace: &Bound,
        project_id: &str,
    ) -> std::result::Result<Value, Fault> {
        let listed = self
            .client
            .json(Method::GET, "/v1/video/projects/private", None, None)
            .await
            .map_err(|_| Fault::new("connection", "Imagine did not respond; retry"))?;
        let path = listed["projects"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|p| {
                p["workspace"]["workspace_id"] == workspace.id.as_str()
                    && p["project_id"] == project_id
            })
            .and_then(|p| p["path"].as_str())
            .filter(|p| Path::new(p).starts_with(&workspace.folder))
            .ok_or_else(|| {
                Fault::new("not_found", "project_id is not a project of this workspace")
            })?
            .to_owned();
        let location = self
            .client
            .json(
                Method::POST,
                "/v1/video/locations",
                Some(&json!({"schema_version":1,"path":path,"create":false})),
                None,
            )
            .await
            .map_err(|_| Fault::new("imagine", "the project could not be reopened"))?;
        self.client
            .json(
                Method::POST,
                "/v1/video/projects/open",
                Some(&json!({"schema_version":1,"location_id":location["location_id"]})),
                None,
            )
            .await
            .map_err(|_| Fault::new("imagine", "the project could not be reopened"))
    }
    /// Turns Imagine paths inside the workspace into file references and
    /// removes every other host path.
    pub(super) fn scrub(&self, value: &mut Value, workspace: Option<&Bound>) {
        match value {
            Value::Object(map) => {
                for key in PATH_KEYS {
                    let Some(text) = map.get(*key).and_then(Value::as_str).map(str::to_owned)
                    else {
                        continue;
                    };
                    if !text.starts_with('/') {
                        continue;
                    }
                    map.remove(*key);
                    let relative = workspace.and_then(|w| {
                        Path::new(&text)
                            .strip_prefix(&w.folder)
                            .ok()
                            .and_then(|r| r.to_str())
                            .map(|r| (w, r.to_owned()))
                    });
                    if let Some((w, relative)) = relative
                        && matches!(*key, "saved_path" | "path")
                        && !map.contains_key("file_id")
                        && Path::new(&text).is_file()
                    {
                        let id = crate::workspace_ids::file_id(&w.id, &relative);
                        map.insert("download_url".into(), json!(self.download_url(&id)));
                        map.insert("file_id".into(), json!(id));
                        if !map.contains_key("name") {
                            map.insert(
                                "name".into(),
                                json!(Path::new(&text).file_name().and_then(|n| n.to_str())),
                            );
                        }
                    }
                }
                for item in map.values_mut() {
                    self.scrub(item, workspace);
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.scrub(item, workspace);
                }
            }
            Value::String(text)
                if text.starts_with(PRIVATE_MOUNT)
                    || workspace
                        .is_some_and(|w| text.starts_with(w.folder.to_string_lossy().as_ref())) =>
            {
                *text = "[hidden]".into();
            }
            _ => {}
        }
    }
    pub(super) async fn workspace_tool(
        &self,
        name: &str,
        args: &Value,
    ) -> std::result::Result<(Value, Vec<Value>), Fault> {
        match name {
            "workspace_create" => {
                let title = args["title"].as_str().unwrap_or("");
                if title.chars().count() > 80 || title.chars().any(char::is_control) {
                    return Err(validation("title must be at most 80 characters"));
                }
                let created = self
                    .client
                    .json(
                        Method::POST,
                        "/v1/workspaces",
                        Some(&json!({"schema_version":1,"title":title})),
                        None,
                    )
                    .await
                    .map_err(validation)?;
                Ok((
                    json!({"workspace_key":created["key"],"title":created["title"],"created_ms":created["created_ms"],
                        "next":"Save workspace_key now where you keep notes for this task. Pass it to every workspace, image and video tool. Imagine cannot list or recover workspaces.",
                        // Every task starts here, so the full rules always arrive.
                        "guide":guide()}),
                    vec![],
                ))
            }
            "workspace_open" => {
                let workspace = self.bound(args).await?;
                let files = self.listing(&workspace).await?;
                let projects = self.workspace_projects(&workspace).await?;
                Ok((
                    json!({"ok":true,"files":files.len(),"video_projects":projects,"guide":guide()}),
                    vec![],
                ))
            }
            "workspace_files" => {
                let workspace = self.bound(args).await?;
                let kind = args["kind"].as_str();
                let files: Vec<Value> = self
                    .listing(&workspace)
                    .await?
                    .into_iter()
                    .filter(|f| kind.is_none_or(|k| f["kind"] == k || f["origin"] == k))
                    .collect();
                Ok((json!({"files":files}), vec![]))
            }
            "workspace_upload" => {
                let workspace = self.bound(args).await?;
                let name = args["name"]
                    .as_str()
                    .filter(|n| !n.is_empty() && n.len() <= 200)
                    .ok_or_else(|| validation("name must be a file name"))?;
                use base64::Engine;
                let data = args["content_base64"]
                    .as_str()
                    .and_then(|d| base64::engine::general_purpose::STANDARD.decode(d).ok())
                    .filter(|d| !d.is_empty())
                    .ok_or_else(|| validation("content_base64 must be base64 file content"))?;
                if data.len() > MAX_TOOL_UPLOAD {
                    return Err(validation(
                        "content is larger than 4 MiB; upload it with HTTP PUT to /v1/workspace/files instead",
                    ));
                }
                let entry = self
                    .put_file(&workspace, name, reqwest::Body::from(data))
                    .await?;
                Ok((self.file_view(&entry), vec![]))
            }
            "workspace_view" => {
                let workspace = self.bound(args).await?;
                let file = file_ident(args)?;
                let side = args["max_side"].as_u64().unwrap_or(1024).clamp(64, 2048) as u32;
                let (path, entry) = self.file_path(&workspace, file).await?;
                if entry["kind"] != "image" {
                    return Err(validation("workspace_view shows images only"));
                }
                let (data, mime) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                    let original = std::fs::read(&path)?;
                    let image = image::load_from_memory(&original)?;
                    let image = if image.width().max(image.height()) > side {
                        image.resize(side, side, image::imageops::FilterType::Lanczos3)
                    } else {
                        image
                    };
                    let mut out = std::io::Cursor::new(vec![]);
                    image
                        .to_rgb8()
                        .write_to(&mut out, image::ImageFormat::Jpeg)?;
                    use base64::Engine;
                    Ok((
                        base64::engine::general_purpose::STANDARD.encode(out.into_inner()),
                        "image/jpeg",
                    ))
                })
                .await
                .map_err(|_| validation("image could not be read"))?
                .map_err(|_| validation("image could not be read or decoded"))?;
                Ok((
                    json!({"file_id":file,"name":entry["name"],"image":0}),
                    vec![json!({"type":"image","data":data,"mimeType":mime})],
                ))
            }
            _ => Err(validation(format!("Unknown tool: {name}"))),
        }
    }
    async fn listing(&self, workspace: &Bound) -> std::result::Result<Vec<Value>, Fault> {
        let listed = self
            .client
            .json(
                Method::GET,
                &format!("/v1/workspaces/{}/files", workspace.id),
                None,
                None,
            )
            .await
            .map_err(|_| Fault::new("connection", "Imagine did not respond; retry"))?;
        Ok(listed["files"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|f| self.file_view(f))
            .collect())
    }
    async fn workspace_projects(
        &self,
        workspace: &Bound,
    ) -> std::result::Result<Vec<Value>, Fault> {
        let listed = self
            .client
            .json(Method::GET, "/v1/video/projects/private", None, None)
            .await
            .map_err(|_| Fault::new("connection", "Imagine did not respond; retry"))?;
        Ok(listed["projects"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|p| p["workspace"]["workspace_id"] == workspace.id.as_str())
            .map(|p| json!({"project_id":p["project_id"],"title":p["folder"],"scenes":p["scene_count"]}))
            .collect())
    }
    async fn put_file(
        &self,
        workspace: &Bound,
        name: &str,
        body: reqwest::Body,
    ) -> std::result::Result<Value, Fault> {
        let response = self
            .client
            .request(
                Method::PUT,
                &format!(
                    "/v1/workspaces/{}/files?name={}",
                    workspace.id,
                    percent_encoding::utf8_percent_encode(name, percent_encoding::NON_ALPHANUMERIC)
                ),
            )
            .timeout(Duration::from_secs(3600))
            .body(body)
            .send()
            .await
            .map_err(|_| Fault::new("connection", "Imagine did not respond; retry"))?;
        let status = response.status();
        let value: Value = response.json().await.unwrap_or(Value::Null);
        if !status.is_success() {
            let code = value["error"]["code"].as_str().unwrap_or("upload_failed");
            return Err(Fault {
                category: "validation".into(),
                message: match code {
                    "upload_type_unsupported" => {
                        "upload a PNG, JPEG, WebP image, or MP4/MOV/WebM video, or WAV/MP3/FLAC/OGG/M4A audio".into()
                    }
                    "upload_too_large" => "the file exceeds the upload limit (images 64 MiB, other media 2 GiB)".into(),
                    _ => format!("Imagine rejected the upload: {code}"),
                },
                code: Some(code.into()),
                at: None,
                detail: None,
            });
        }
        Ok(value)
    }
}

// HTTP file routes on the MCP port. The workspace key is the bearer token.
fn bearer(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers
        .get("authorization")?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .filter(|key| crate::workspace_ids::key_shaped(key))
}
fn http_error(status: u16, code: &str) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(json!({"error":{"code":code}}).to_string()))
        .unwrap()
}
async fn http_bound(
    mcp: &Mcp,
    headers: &axum::http::HeaderMap,
) -> std::result::Result<(Mcp, Bound), Response<Body>> {
    // Like /mcp: only this machine's own name for the port, against DNS rebinding.
    let port = mcp.inner.config.mcp_port.unwrap_or(0);
    let host = headers.get("host").and_then(|v| v.to_str().ok());
    if !host.is_some_and(|h| {
        [format!("127.0.0.1:{port}"), format!("localhost:{port}")]
            .iter()
            .any(|v| v == h)
    }) {
        return Err(http_error(421, "invalid_host"));
    }
    if mcp.refresh_session().await.is_err() {
        return Err(http_error(503, "imagine_unavailable"));
    }
    let active = mcp.active();
    let key = bearer(headers).ok_or_else(|| http_error(401, "workspace_key_required"))?;
    let workspace = active
        .resolve_key(key)
        .await
        .map_err(|_| http_error(404, "workspace_unknown"))?;
    Ok((active, workspace))
}
pub(super) async fn list_files(
    State(mcp): State<Mcp>,
    headers: axum::http::HeaderMap,
) -> Response<Body> {
    let (active, workspace) = match http_bound(&mcp, &headers).await {
        Ok(bound) => bound,
        Err(response) => return response,
    };
    match active.listing(&workspace).await {
        Ok(files) => axum::Json(json!({"files":files})).into_response(),
        Err(_) => http_error(503, "imagine_unavailable"),
    }
}
#[derive(serde::Deserialize)]
pub(super) struct UploadName {
    #[serde(default)]
    name: String,
}
pub(super) async fn upload_file(
    State(mcp): State<Mcp>,
    axum::extract::Query(query): axum::extract::Query<UploadName>,
    headers: axum::http::HeaderMap,
    body: Body,
) -> Response<Body> {
    let (active, workspace) = match http_bound(&mcp, &headers).await {
        Ok(bound) => bound,
        Err(response) => return response,
    };
    let stream = body.into_data_stream();
    match active
        .put_file(&workspace, &query.name, reqwest::Body::wrap_stream(stream))
        .await
    {
        Ok(entry) => (StatusCode::CREATED, axum::Json(active.file_view(&entry))).into_response(),
        Err(fault) => http_error(
            if fault.code.as_deref() == Some("upload_too_large") {
                413
            } else {
                422
            },
            fault.code.as_deref().unwrap_or("upload_failed"),
        ),
    }
}
pub(super) async fn download_file(
    State(mcp): State<Mcp>,
    UrlPath(file): UrlPath<String>,
    headers: axum::http::HeaderMap,
) -> Response<Body> {
    let (active, workspace) = match http_bound(&mcp, &headers).await {
        Ok(bound) => bound,
        Err(response) => return response,
    };
    if file_ident(&json!({"file_id":file})).is_err() {
        return http_error(404, "file_not_found");
    }
    let mut request = active
        .client
        .request(
            Method::GET,
            &format!("/v1/workspaces/{}/files/{file}", workspace.id),
        )
        .timeout(Duration::from_secs(3600));
    for name in ["range", "if-range"] {
        if let Some(value) = headers.get(name) {
            request = request.header(name, value);
        }
    }
    let Ok(response) = request.send().await else {
        return http_error(503, "imagine_unavailable");
    };
    let mut builder = Response::builder().status(response.status().as_u16());
    for name in [
        "content-type",
        "content-length",
        "content-range",
        "accept-ranges",
        "etag",
        "content-disposition",
    ] {
        if let Some(value) = response.headers().get(name) {
            builder = builder.header(name, value.as_bytes());
        }
    }
    builder
        .body(Body::from_stream(response.bytes_stream()))
        .unwrap_or_else(|_| http_error(500, "download_failed"))
}
