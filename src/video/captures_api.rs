use super::*;

pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/projects/{id}/captures", post(mutate))
        .route("/v1/video/projects/{id}/captures/import", post(import))
        .route("/v1/video/projects/{id}/takes/cleanup", post(cleanup))
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Cleanup {
    schema_version: u32,
    expected_project_revision: u64,
    /// False previews the counts; true clears at the previewed revision.
    apply: bool,
}

/// Director's project Take cleanup: preview, then apply at the same revision.
async fn cleanup(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Cleanup>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let id = ProjectId(id);
    let staging = staging(&app)?;
    let mut service = app.video.lock().await;
    if !request.apply {
        let session = service.projects.session(&id)?;
        if session.document().project.project_revision != request.expected_project_revision {
            return Err(StorageError::new("revision_conflict").into());
        }
        let preview = session.cleanup_preview()?;
        return Ok(Json(json!({"schema_version":1,"preview":preview})));
    }
    let warning = match service
        .projects
        .session_mut(&id)?
        .clear_takes(request.expected_project_revision, &staging)
    {
        Ok(warning) => warning,
        Err(error) => {
            let mut error = Error::from(error);
            error.snapshot = service.snapshot(&id).ok().map(Box::new);
            return Err(error);
        }
    };
    service.runtime_takes.remove(&id);
    let mut reply = serde_json::to_value(service.snapshot(&id)?).map_err(internal)?;
    reply["cleanup_result"] = json!({"cleanup_warning": warning});
    Ok(Json(reply))
}

async fn import(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<crate::video::captures::Import>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let id = ProjectId(id);
    let file = {
        let service = app.video.lock().await;
        let session = service.projects.session(&id)?;
        if session.document().project.project_revision != request.expected_project_revision {
            return Err(StorageError::new("revision_conflict").into());
        }
        let asset = session
            .document()
            .project
            .assets
            .get(&request.asset_id)
            .ok_or_else(|| media_error("asset_not_found"))?;
        let resource = session.document().resources.iter().find(|r| matches!(&r.kind,ResourceKind::Reference{asset_id,revision} if asset_id == &asset.id && *revision == asset.revision)).ok_or_else(|| media_error("reference_resource_missing"))?;
        session.resource_file(&resource.id)?
    };
    let info = tokio::task::spawn_blocking(move || media::detect_streamed(&file))
        .await
        .map_err(internal)?
        .map_err(media_error)?;
    if info.kind != MediaKind::Video {
        return Err(media_error("take_requires_video"));
    }
    let staging = staging(&app)?;
    let mut service = app.video.lock().await;
    let change = match service
        .projects
        .session_mut(&id)?
        .import_take(request.clone(), &staging)
    {
        Ok(change) => change,
        Err(error) => {
            let mut error = Error::from(error);
            error.snapshot = service.snapshot(&id).ok().map(Box::new);
            return Err(error);
        }
    };
    if let Some(asset) = service
        .runtime_assets
        .get(&id)
        .and_then(|m| m.get(&request.asset_id))
        .cloned()
    {
        service
            .runtime_takes
            .entry(id.clone())
            .or_default()
            .insert(change.attached_id.clone().unwrap(), asset);
    }
    let mut reply = serde_json::to_value(service.snapshot(&id)?).map_err(internal)?;
    reply["capture_result"] = json!({"attached_id":change.attached_id,"imported":true});
    Ok(Json(reply))
}

async fn mutate(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<crate::video::captures::Request>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    let source = request.take_id.clone();
    let id = ProjectId(id);
    let staging = staging(&app)?;
    let mut service = app.video.lock().await;
    let change = match service
        .projects
        .session_mut(&id)?
        .capture(request, &staging)
    {
        Ok(change) => change,
        Err(error) => {
            let mut error = Error::from(error);
            error.snapshot = service.snapshot(&id).ok().map(Box::new);
            return Err(error);
        }
    };
    if let Some(new_id) = change.attached_id.as_ref()
        && let Some(asset) = service
            .runtime_takes
            .get(&id)
            .and_then(|m| m.get(&source))
            .cloned()
    {
        service
            .runtime_takes
            .entry(id.clone())
            .or_default()
            .insert(new_id.clone(), asset);
    }
    if let Some(bindings) = service.runtime_takes.get_mut(&id) {
        bindings.retain(|id, _| change.document.takes.iter().any(|take| &take.id == id));
    }
    let mut reply = serde_json::to_value(service.snapshot(&id)?).map_err(internal)?;
    reply["capture_result"] = json!({"attached_id": change.attached_id, "retained_for_provenance":change.retained_for_provenance, "cleanup_warning":change.cleanup_warning});
    Ok(Json(reply))
}
