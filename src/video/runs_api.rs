use super::*;
use crate::scheduler::Action;
use axum::http::HeaderMap;
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/runs", post(start).get(all))
        .route("/v1/video/runs/plan", post(plan))
        .route("/v1/video/runs/{id}", get(read))
        .route("/v1/video/runs/{id}/stop", post(stop))
        .route("/v1/video/projects/{id}/runs", get(list))
}
async fn all(State(app): State<App>) -> Result<Json<Value>, crate::scheduler::Fault> {
    Ok(Json(
        app.control
            .call(Action::ReadVideoRuns {
                owner: "local".into(),
                project_id: None,
                id: None,
            })
            .await?,
    ))
}
async fn start(
    State(app): State<App>,
    headers: HeaderMap,
    body: Result<Json<super::super::runs::Request>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), crate::scheduler::Fault> {
    let Json(request) = body.map_err(|_| crate::scheduler::Fault::new(422, "invalid_request"))?;
    let key = headers
        .get("Idempotency-Key")
        .and_then(|h| h.to_str().ok())
        .ok_or(crate::scheduler::Fault::new(
            400,
            "idempotency_key_required",
        ))?;
    let value = app
        .control
        .call(Action::StartVideoRun {
            owner: "local".into(),
            request,
            key: key.into(),
        })
        .await?;
    Ok((StatusCode::ACCEPTED, Json(value)))
}
async fn plan(
    State(app): State<App>,
    body: Result<Json<super::super::runs::Request>, JsonRejection>,
) -> Result<Json<Value>, crate::scheduler::Fault> {
    let Json(request) = body.map_err(|_| crate::scheduler::Fault::new(422, "invalid_request"))?;
    Ok(Json(
        app.control
            .call(Action::PlanVideoRun {
                owner: "local".into(),
                request,
            })
            .await?,
    ))
}
async fn read(
    State(app): State<App>,
    Path(id): Path<String>,
) -> Result<Json<Value>, crate::scheduler::Fault> {
    Ok(Json(
        app.control
            .call(Action::ReadVideoRuns {
                owner: "local".into(),
                project_id: None,
                id: Some(id),
            })
            .await?,
    ))
}
async fn list(
    State(app): State<App>,
    Path(id): Path<String>,
) -> Result<Json<Value>, crate::scheduler::Fault> {
    Ok(Json(
        app.control
            .call(Action::ReadVideoRuns {
                owner: "local".into(),
                project_id: Some(ProjectId(id)),
                id: None,
            })
            .await?,
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Stop {
    schema_version: u32,
    cancel_current: bool,
}
async fn stop(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Stop>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    app.control
        .call(Action::StopVideoRun {
            owner: "local".into(),
            id,
            cancel_current: request.cancel_current,
        })
        .await
        .map(Json)
        .map_err(|e| Error::new(StatusCode::from_u16(e.status).unwrap(), e.code))
}
