use crate::{
    config::Access,
    contract::{self},
    engine::EngineState,
    scheduler::{Action, Control, Fault},
    storage::{Store, hash},
};
use axum::{
    Extension, Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Request, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{convert::Infallible, sync::Arc, time::Duration};
use subtle::ConstantTimeEq;
use tokio::sync::{Semaphore, watch};

#[derive(Clone)]
pub struct App {
    pub profiles: Arc<crate::profiles::Catalog>,
    pub store: Store,
    pub control: Control,
    pub engine: watch::Receiver<EngineState>,
    pub access: Arc<Access>,
    pub uploads: Arc<Semaphore>,
}
#[derive(Clone)]
struct Principal {
    owner: String,
    admin: bool,
}
impl IntoResponse for Fault {
    fn into_response(self) -> Response {
        (
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(json!({"error":{"code":self.code}})),
        )
            .into_response()
    }
}
pub fn router(app: App) -> Router {
    Router::new()
        .route("/v1/session", get(session))
        .route("/v1/profiles", get(profiles))
        .route("/v1/profiles/{id}", get(profile))
        .route("/v1/jobs/validate", post(validate_job))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model))
        .route("/v1/assets", post(upload))
        .route("/v1/assets/{id}", get(download))
        .route("/v1/jobs", post(submit))
        .route("/v1/jobs/batch", post(submit_batch))
        .route("/v1/jobs/{id}", get(job))
        .route("/v1/jobs/{id}/events", get(events))
        .route("/v1/jobs/{id}/cancel", post(cancel))
        .route("/v1/runtime", get(runtime))
        .route("/v1/runtime/unload", post(unload))
        .route("/v1/runtime/resume", post(resume))
        .route("/v1/runtime/yield", post(yield_gpu))
        .route("/v1/operations/{id}", get(operation))
        .fallback(|| async { Fault::new(404, "not_found") })
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(app.clone(), authenticate))
        .with_state(app)
}
async fn authenticate(State(app): State<App>, mut request: Request, next: Next) -> Response {
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let principal = token.and_then(|token| {
        let supplied = hash(token.as_bytes());
        if bool::from(
            supplied
                .as_bytes()
                .ct_eq(hash(app.access.admin_token.as_bytes()).as_bytes()),
        ) {
            return Some(Principal {
                owner: "admin".into(),
                admin: true,
            });
        }
        app.access.clients.iter().find_map(|(owner, token)| {
            bool::from(supplied.as_bytes().ct_eq(hash(token.as_bytes()).as_bytes())).then(|| {
                Principal {
                    owner: owner.clone(),
                    admin: false,
                }
            })
        })
    });
    let mut response = if let Some(principal) = principal {
        if request
            .headers()
            .get("X-Imagine-Session")
            .is_some_and(|id| id.as_bytes() != app.store.session.id.as_bytes())
        {
            Fault::new(409, "session_changed").into_response()
        } else {
            request.extensions_mut().insert(principal);
            next.run(request).await
        }
    } else {
        Fault::new(401, "unauthorized").into_response()
    };
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
        .headers_mut()
        .insert(header::X_CONTENT_TYPE_OPTIONS, "nosniff".parse().unwrap());
    response
}
async fn session(State(app): State<App>) -> Json<Value> {
    // Preserve the response shape for clients that inspect the storage mode.
    Json(json!({"id": app.store.session.id, "encrypted": false}))
}
async fn models() -> Json<Value> {
    Json(contract::models())
}
async fn model(Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    contract::model(&id)
        .map(Json)
        .ok_or(Fault::new(404, "model_not_found"))
}
async fn upload(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, Json<Value>), Fault> {
    let mime = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !["image/png", "image/jpeg", "image/webp"].contains(&mime) {
        return Err(Fault::new(415, "unsupported_media_type"));
    }
    let _permit = app
        .uploads
        .clone()
        .try_acquire_owned()
        .map_err(|_| Fault::new(429, "upload_busy"))?;
    let asset = app
        .store
        .save_asset(&p.owner, body.to_vec())
        .await
        .map_err(|_| Fault::new(422, "invalid_image"))?;
    Ok((StatusCode::CREATED, Json(asset)))
}
async fn download(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Result<Response, Fault> {
    let info = app
        .store
        .asset(&id, &p.owner)
        .await?
        .ok_or(Fault::new(404, "asset_not_found"))?;
    let bytes = app.store.read_asset(&id, &p.owner).await?;
    Ok((
        [
            (header::CONTENT_TYPE, info["mime_type"].as_str().unwrap()),
            (header::CONTENT_DISPOSITION, "attachment"),
        ],
        bytes,
    )
        .into_response())
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchRequest {
    jobs: Vec<Value>,
}
async fn submit_batch(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    headers: HeaderMap,
    body: Result<Json<BatchRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Fault> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let key = headers
        .get("Idempotency-Key")
        .map(|v| v.to_str().map(str::to_owned))
        .transpose()
        .map_err(|_| Fault::new(400, "invalid_idempotency_key"))?;
    let mut requests = Vec::with_capacity(request.jobs.len());
    for request in request.jobs {
        requests.push(app.profiles.resolve(request, &app.store, &p.owner).await?);
    }
    let result = app
        .control
        .call(Action::SubmitBatch {
            owner: p.owner,
            requests,
            key,
        })
        .await?;
    Ok((StatusCode::ACCEPTED, Json(result)))
}
async fn submit(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Fault> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let key = headers
        .get("Idempotency-Key")
        .map(|v| v.to_str().map(str::to_owned))
        .transpose()
        .map_err(|_| Fault::new(400, "invalid_idempotency_key"))?;
    let request = app.profiles.resolve(request, &app.store, &p.owner).await?;
    let mut result = app
        .control
        .call(Action::Submit {
            owner: p.owner,
            request: Box::new(request),
            key,
        })
        .await?;
    let id = result["id"].as_str().unwrap().to_owned();
    result["status_url"] = json!(format!("/v1/jobs/{id}"));
    result["events_url"] = json!(format!("/v1/jobs/{id}/events"));
    Ok((StatusCode::ACCEPTED, Json(result)))
}
async fn job(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Fault> {
    Ok(Json(job_snapshot(&app, &id, &p.owner).await?))
}
async fn job_snapshot(app: &App, id: &str, owner: &str) -> Result<Value, Fault> {
    let mut value = app
        .store
        .job(id, owner)
        .await?
        .ok_or(Fault::new(404, "job_not_found"))?;
    value["progress"] = Value::Null;
    let state = app.engine.borrow();
    if matches!(value["status"].as_str(), Some("running" | "cancelling"))
        && state.job_id.as_deref() == Some(id)
    {
        value["phase"] = json!(state.phase);
        value["stage"] = json!(state.stage);
        value["progress"] = json!(state.progress);
    }
    Ok(value)
}
async fn cancel(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<Value>), Fault> {
    Ok((
        StatusCode::ACCEPTED,
        Json(
            app.control
                .call(Action::Cancel { owner: p.owner, id })
                .await?,
        ),
    ))
}
async fn events(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Result<Response, Fault> {
    if app.store.job(&id, &p.owner).await?.is_none() {
        return Err(Fault::new(404, "job_not_found"));
    }
    let stream = async_stream::stream! {
        let mut last=String::new();
        loop {
            let Ok(value)=job_snapshot(&app,&id,&p.owner).await else {break};
            let terminal=matches!(value["status"].as_str(),Some("succeeded"|"failed"|"cancelled"|"interrupted"));
            let data=value.to_string();
            if data!=last {yield Ok::<Event,Infallible>(Event::default().event("job").data(&data));last=data;}
            if terminal{break;}
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    };
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}
async fn runtime(State(app): State<App>) -> Json<Value> {
    let mut value = app.control.state.borrow().clone();
    value["engine"] = json!(*app.engine.borrow());
    value["api_revision"] = json!(contract::API_REVISION);
    Json(value)
}
fn admin(p: &Principal) -> Result<(), Fault> {
    if p.admin {
        Ok(())
    } else {
        Err(Fault::new(403, "admin_required"))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Unload {
    active_job: UnloadMode,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnloadMode {
    Wait,
    Cancel,
}
async fn unload(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    body: Result<Json<Unload>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Fault> {
    admin(&p)?;
    let Json(body) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let mut value = app
        .control
        .call(Action::Unload {
            cancel: matches!(body.active_job, UnloadMode::Cancel),
        })
        .await?;
    value["status_url"] = json!(format!("/v1/operations/{}", value["id"].as_str().unwrap()));
    Ok((StatusCode::ACCEPTED, Json(value)))
}
async fn resume(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
) -> Result<Json<Value>, Fault> {
    admin(&p)?;
    Ok(Json(app.control.call(Action::Resume).await?))
}
async fn operation(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Fault> {
    admin(&p)?;
    Ok(Json(
        app.store
            .operation(&id)
            .await?
            .ok_or(Fault::new(404, "operation_not_found"))?,
    ))
}

async fn yield_gpu(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
) -> Result<Json<Value>, Fault> {
    admin(&p)?;
    Ok(Json(app.control.call(Action::YieldGpu).await?))
}

async fn profiles(State(app): State<App>) -> Json<Value> {
    Json(app.profiles.list())
}
async fn profile(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    app.profiles.describe(&id).map(Json)
}
async fn validate_job(
    State(app): State<App>,
    Extension(p): Extension<Principal>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, Fault> {
    let Json(value) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let request = app.profiles.resolve(value, &app.store, &p.owner).await?;
    crate::generation::reference_metadata(&request, &p.owner, &app.store).await?;
    let mut effective = request.effective();
    // Validation is safe to expose in MCP results; prompt text stays private.
    for key in ["prompt", "negative_prompt", "client_metadata"] {
        effective.as_object_mut().unwrap().remove(key);
    }
    if let Some(sheet) = effective
        .get_mut("character_sheet")
        .and_then(Value::as_object_mut)
    {
        sheet.remove("character_description");
        sheet.remove("clothing_description");
    }
    Ok(Json(json!({"valid":true,"effective":effective})))
}
