use super::*;
impl From<Fault> for Error {
    fn from(fault: Fault) -> Self {
        Error::new(
            StatusCode::from_u16(fault.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            fault.code,
        )
    }
}
use crate::scheduler::Fault;
use imagine_video_contract::authoring::{Request as AuthorRequest, Review};
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/authoring/profiles", get(profiles))
        .route("/v1/video/projects/{id}/authoring", post(start))
        .route("/v1/video/authoring/{id}", get(read))
        .route("/v1/video/authoring/{id}/review", post(review))
        .route("/v1/video/authoring/{id}/expand", post(expand))
        .route("/v1/video/authoring/{id}/cancel", post(cancel))
        .route("/v1/video/authoring/{id}/retry", post(retry))
        .route("/v1/video/authoring/{id}/apply", post(apply))
}
async fn profiles() -> Result<Json<Value>, Fault> {
    let catalog = super::super::authoring::profiles::Catalog::load(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/video/prompt-studio/profiles"
    )))
    .map_err(|c| Fault::new(422, c))?;
    Ok(Json(json!(catalog)))
}
async fn registry(app: &App) -> std::sync::Arc<super::super::authoring::drafts::Service> {
    app.video.lock().await.authoring.clone()
}
async fn start(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<AuthorRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Error> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let provider = app.enhancement.snapshot(&request.provider_profile_id)?;
    let (base, authoring, grounding) = {
        let service = app.video.lock().await;
        let session = service.projects.session(&ProjectId(id.clone()))?;
        let grounding = super::super::authoring::grounding::Grounding::capture(session, &request)
            .map_err(media_error)?;
        (
            service.projects.read(&ProjectId(id)).map_err(Error::from)?,
            service.authoring.clone(),
            grounding,
        )
    };
    let draft = authoring.start(base, request, provider, app.enhancement.clone(), grounding)?;
    Ok((StatusCode::ACCEPTED, Json(json!(draft))))
}
async fn read(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    let service = registry(&app).await;
    let mut value = json!(service.read(&id)?);
    value["progress"] = service.progress(&id)?;
    Ok(Json(value))
}
async fn cancel(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    let service = app.video.lock().await;
    Ok(Json(json!(service.authoring.cancel(&id)?)))
}
async fn retry(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    Ok(Json(json!(
        registry(&app).await.retry(&id, app.enhancement.clone())?
    )))
}
async fn review(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Review>, JsonRejection>,
) -> Result<Json<Value>, Fault> {
    let Json(review) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let service = app.video.lock().await;
    Ok(Json(json!(service.authoring.review(&id, review)?)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expand {
    schema_version: u32,
    start_index: usize,
}
async fn expand(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Expand>, JsonRejection>,
) -> Result<Json<Value>, Fault> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    if request.schema_version != 1 {
        return Err(Fault::new(422, "unsupported_schema"));
    }
    let service = app.video.lock().await;
    Ok(Json(json!(service.authoring.expand(
        &id,
        request.start_index,
        app.enhancement.clone()
    )?)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Apply {
    schema_version: u32,
    expected_project_revision: u64,
}
async fn apply(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Apply>, JsonRejection>,
) -> Result<Json<Snapshot>, Error> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    if request.schema_version != 1 {
        return Err(Fault::new(422, "unsupported_schema").into());
    }
    let mut service = app.video.lock().await;
    let draft = service.authoring.read(&id)?;
    let current = service
        .projects
        .read(&draft.project_id)
        .map_err(Error::from)?;
    if request.expected_project_revision != current.project_revision {
        return Err(Fault::new(409, "authoring_revision_conflict").into());
    }
    let plan = service.authoring.edit_plan(&id, &current)?;
    service
        .projects
        .session_mut(&current.id)
        .map_err(Error::from)?
        .apply_authoring(plan, draft)
        .map_err(Error::from)?;
    service.authoring.applied(&id);
    Ok(Json(service.snapshot(&current.id).map_err(Error::from)?))
}
