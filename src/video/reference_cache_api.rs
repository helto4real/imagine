use super::super::reference_cache::{Control, Status};
use super::*;
use std::sync::Arc;
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/reference-cache", get(status).post(configure))
        .route("/v1/video/reference-cache/clear", post(clear))
}
async fn owner(app: &App) -> Option<Arc<dyn Control>> {
    app.video.lock().await.reference_cache.clone()
}
async fn status(State(app): State<App>) -> Result<Json<Value>, Error> {
    let status = match owner(&app).await {
        Some(c) => c.status().map_err(media_error)?,
        None => Status::unavailable(),
    };
    Ok(Json(json!(status)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    schema_version: u32,
    limit_bytes: u64,
    enabled: bool,
}
async fn configure(
    State(app): State<App>,
    body: Result<Json<Policy>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let p = input(body)?;
    schema(p.schema_version)?;
    let c = owner(&app)
        .await
        .ok_or_else(|| media_error("reference_cache_unavailable"))?;
    Ok(Json(json!(
        c.configure(p.limit_bytes, p.enabled).map_err(media_error)?
    )))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}
async fn clear(
    State(app): State<App>,
    body: Result<Json<Empty>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    input(body)?;
    let c = owner(&app)
        .await
        .ok_or_else(|| media_error("reference_cache_unavailable"))?;
    Ok(Json(json!(c.clear().map_err(media_error)?)))
}
