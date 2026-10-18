use crate::{
    contract::{self},
    engine::EngineState,
    scheduler::{Action, Control, Fault},
    storage::Store,
};
use axum::{
    Json, Router,
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
use tokio::sync::{Semaphore, watch};

#[derive(Clone)]
pub struct App {
    pub library: Arc<std::sync::Mutex<crate::model_library::Library>>,
    pub profiles: Arc<crate::profiles::Catalog>,
    pub store: Store,
    pub control: Control,
    pub engine: watch::Receiver<EngineState>,
    pub uploads: Arc<Semaphore>,
}
const LOCAL_OWNER: &str = "local";

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
        .route(
            "/v1/model-library",
            get(model_library).put(update_model_library),
        )
        .route("/v1/model-library/refresh", post(refresh_model_library))
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
        .layer(middleware::from_fn_with_state(app.clone(), session_guard))
        .with_state(app)
}
async fn session_guard(State(app): State<App>, request: Request, next: Next) -> Response {
    let mut response = if request
        .headers()
        .get("X-Imagine-Session")
        .is_some_and(|id| id.as_bytes() != app.store.session.id.as_bytes())
    {
        Fault::new(409, "session_changed").into_response()
    } else {
        next.run(request).await
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
async fn models(State(app): State<App>) -> Json<Value> {
    let mut value = contract::models();
    let backend = app.engine.borrow().backend.clone();
    for model in value["models"].as_array_mut().unwrap() {
        *model = crate::comfy_engine::model_metadata(model.take(), &backend);
    }
    Json(value)
}
async fn model(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    contract::model(&id)
        .map(|model| crate::comfy_engine::model_metadata(model, &app.engine.borrow().backend))
        .map(Json)
        .ok_or(Fault::new(404, "model_not_found"))
}
async fn upload(
    State(app): State<App>,
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
        .save_asset(LOCAL_OWNER, body.to_vec())
        .await
        .map_err(|_| Fault::new(422, "invalid_image"))?;
    Ok((StatusCode::CREATED, Json(asset)))
}
async fn download(State(app): State<App>, Path(id): Path<String>) -> Result<Response, Fault> {
    let info = app
        .store
        .asset(&id, LOCAL_OWNER)
        .await?
        .ok_or(Fault::new(404, "asset_not_found"))?;
    let bytes = app.store.read_asset(&id, LOCAL_OWNER).await?;
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
        requests.push(resolve_request(&app, request).await?);
    }
    let result = app
        .control
        .call(Action::SubmitBatch {
            owner: LOCAL_OWNER.into(),
            requests,
            key,
        })
        .await?;
    Ok((StatusCode::ACCEPTED, Json(result)))
}
async fn submit(
    State(app): State<App>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Fault> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let key = headers
        .get("Idempotency-Key")
        .map(|v| v.to_str().map(str::to_owned))
        .transpose()
        .map_err(|_| Fault::new(400, "invalid_idempotency_key"))?;
    let request = resolve_request(&app, request).await?;
    let mut result = app
        .control
        .call(Action::Submit {
            owner: LOCAL_OWNER.into(),
            request: Box::new(request),
            key,
        })
        .await?;
    let id = result["id"].as_str().unwrap().to_owned();
    result["status_url"] = json!(format!("/v1/jobs/{id}"));
    result["events_url"] = json!(format!("/v1/jobs/{id}/events"));
    Ok((StatusCode::ACCEPTED, Json(result)))
}
async fn job(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    Ok(Json(job_snapshot(&app, &id, LOCAL_OWNER).await?))
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
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<Value>), Fault> {
    Ok((
        StatusCode::ACCEPTED,
        Json(
            app.control
                .call(Action::Cancel {
                    owner: LOCAL_OWNER.into(),
                    id,
                })
                .await?,
        ),
    ))
}
async fn events(State(app): State<App>, Path(id): Path<String>) -> Result<Response, Fault> {
    if app.store.job(&id, LOCAL_OWNER).await?.is_none() {
        return Err(Fault::new(404, "job_not_found"));
    }
    let stream = async_stream::stream! {
        let mut last=String::new();
        loop {
            let Ok(value)=job_snapshot(&app,&id,LOCAL_OWNER).await else {break};
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
    body: Result<Json<Unload>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Fault> {
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
async fn resume(State(app): State<App>) -> Result<Json<Value>, Fault> {
    Ok(Json(app.control.call(Action::Resume).await?))
}
async fn operation(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    Ok(Json(
        app.store
            .operation(&id)
            .await?
            .ok_or(Fault::new(404, "operation_not_found"))?,
    ))
}

async fn yield_gpu(State(app): State<App>) -> Result<Json<Value>, Fault> {
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
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, Fault> {
    let Json(value) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let request = resolve_request(&app, value).await?;
    crate::generation::reference_metadata(&request, LOCAL_OWNER, &app.store).await?;
    let mut effective = request.effective();
    // Validation is safe to expose in MCP results; prompt text stays private.
    for key in [
        "prompt",
        "negative_prompt",
        "client_metadata",
        "resolved_models",
    ] {
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

async fn resolve_request(app: &App, value: Value) -> Result<crate::contract::JobRequest, Fault> {
    let library = app.library.clone();
    let (value, resolved) = tokio::task::spawn_blocking(move || {
        library
            .lock()
            .map_err(|_| Fault::new(500, "model_library_unavailable"))?
            .prepare(value)
    })
    .await
    .map_err(|_| Fault::new(500, "model_library_unavailable"))??;
    let mut request = app.profiles.resolve(value, &app.store, LOCAL_OWNER).await?;
    crate::engine::validate_selection(
        &app.engine.borrow().backend,
        &request.model,
        &request.sampling,
    )
    .map_err(|e| {
        Fault::new(
            422,
            e.downcast_ref::<crate::engine::EngineError>()
                .map_or("unsupported_sampling", |e| e.0),
        )
    })?;
    request.resolved_models = if resolved.is_some() {
        resolved
    } else {
        let library = app.library.clone();
        let request = request.clone();
        tokio::task::spawn_blocking(move || {
            library
                .lock()
                .map_err(|_| Fault::new(500, "model_library_unavailable"))?
                .pin_explicit(&request)
        })
        .await
        .map_err(|_| Fault::new(500, "model_library_unavailable"))??
    };
    Ok(request)
}
async fn model_library(
    State(app): State<App>,
) -> Result<Json<crate::model_library::LibrarySnapshot>, Fault> {
    library_action(app, None, false).await
}
async fn refresh_model_library(
    State(app): State<App>,
) -> Result<Json<crate::model_library::LibrarySnapshot>, Fault> {
    library_action(app, None, true).await
}
async fn update_model_library(
    State(app): State<App>,
    body: Result<Json<crate::model_library::LibrarySettings>, JsonRejection>,
) -> Result<Json<crate::model_library::LibrarySnapshot>, Fault> {
    let Json(settings) = body.map_err(|_| Fault::new(422, "invalid_library_settings"))?;
    library_action(app, Some(settings), false).await
}
async fn library_action(
    app: App,
    settings: Option<crate::model_library::LibrarySettings>,
    refresh: bool,
) -> Result<Json<crate::model_library::LibrarySnapshot>, Fault> {
    tokio::task::spawn_blocking(move || {
        let mut library = app
            .library
            .lock()
            .map_err(|_| Fault::new(500, "model_library_unavailable"))?;
        if let Some(settings) = settings {
            library
                .update(settings)
                .map_err(|_| Fault::new(422, "model_library_update_failed"))?;
        } else if refresh {
            library
                .scan()
                .map_err(|_| Fault::new(422, "model_library_scan_failed"))?;
        }
        let mut snapshot = library.snapshot();
        let backend = app.engine.borrow().backend.clone();
        for model in &mut snapshot.models {
            *model = crate::comfy_engine::model_metadata(model.take(), &backend);
        }
        Ok(Json(snapshot))
    })
    .await
    .map_err(|_| Fault::new(500, "model_library_unavailable"))?
}
