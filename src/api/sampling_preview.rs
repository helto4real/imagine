//! Session-bound latest-preview API. Never registers media, reads pixels from disk,
//! or includes pixels in runtime/job JSON. See binary_response for framing.
use super::{App, Fault, LOCAL_OWNER};
use crate::{
    h3_protocol::preview::Settings,
    sampling_preview::{Clip, Hub, Status},
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{
        Path, Query, Request, State,
        rejection::{JsonRejection, QueryRejection},
    },
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Deserialize;
use std::{convert::Infallible, sync::Arc};

pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/runtime/sampling-preview", get(status).post(configure))
        .route("/v1/jobs/{id}/sampling-preview", get(latest))
        .layer(middleware::from_fn(no_store))
}
async fn no_store(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
        .headers_mut()
        .insert(header::X_CONTENT_TYPE_OPTIONS, "nosniff".parse().unwrap());
    response
}
fn session(app: &App, headers: &HeaderMap) -> Result<(), Fault> {
    let id = headers
        .get("X-Imagine-Session")
        .ok_or(Fault::new(401, "session_required"))?;
    if id.as_bytes() != app.store.session.id.as_bytes() {
        return Err(Fault::new(409, "session_changed"));
    }
    Ok(())
}
fn hub(app: &App) -> Arc<Hub> {
    app.engine.borrow().sampling_preview.clone()
}
async fn status(State(app): State<App>, headers: HeaderMap) -> Result<Json<Status>, Fault> {
    session(&app, &headers)?;
    Ok(Json(hub(&app).status()))
}
async fn configure(
    State(app): State<App>,
    headers: HeaderMap,
    body: Result<Json<Settings>, JsonRejection>,
) -> Result<Json<Status>, Fault> {
    session(&app, &headers)?;
    let Json(settings) = body.map_err(|_| Fault::new(422, "invalid_sampling_preview_settings"))?;
    let hub = hub(&app);
    hub.configure(settings)
        .map_err(|code| Fault::new(422, code))?;
    Ok(Json(hub.status()))
}
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct After {
    after: u64,
}
async fn latest(
    State(app): State<App>,
    headers: HeaderMap,
    Path(id): Path<String>,
    query: Result<Query<After>, QueryRejection>,
) -> Result<Response, Fault> {
    session(&app, &headers)?;
    let Query(query) = query.map_err(|_| Fault::new(422, "invalid_request"))?;
    // Same owned job lookup as /v1/jobs/{id}; never reveal another owner's pixels.
    let Some(job) = app.store.job(&id, LOCAL_OWNER).await? else {
        return Err(Fault::new(404, "job_not_found"));
    };
    let hub = hub(&app);
    if !matches!(job["status"].as_str(), Some("running" | "cancelling")) {
        if matches!(
            job["status"].as_str(),
            Some("succeeded" | "failed" | "cancelled" | "interrupted")
        ) {
            hub.end(&id);
        }
        return Ok(StatusCode::NO_CONTENT.into_response());
    }
    let Some(clip) = hub.snapshot(&id, query.after) else {
        return Ok(StatusCode::NO_CONTENT.into_response());
    };
    binary_response(clip)
}

struct Payload(Arc<Clip>);
impl AsRef<[u8]> for Payload {
    fn as_ref(&self) -> &[u8] {
        &self.0.bytes
    }
}
/// Atomic full-clip framing: u32 big-endian JSON byte length, Metadata JSON, RGB8.
/// Arc-backed Bytes avoids an additional full-payload copy. An in-flight response
/// may retain its snapshot until sent, while Hub owns only its newest clip.
fn binary_response(clip: Arc<Clip>) -> Result<Response, Fault> {
    let metadata =
        serde_json::to_vec(&clip.metadata).map_err(|_| Fault::new(500, "internal_error"))?;
    let prefix = u32::try_from(metadata.len())
        .map_err(|_| Fault::new(500, "internal_error"))?
        .to_be_bytes();
    let size = 4 + metadata.len() + clip.bytes.len();
    let chunks = [
        Bytes::copy_from_slice(&prefix),
        Bytes::from(metadata),
        Bytes::from_owner(Payload(clip)),
    ];
    let stream = futures_util::stream::iter(chunks.into_iter().map(Ok::<_, Infallible>));
    let mut response = Body::from_stream(stream).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        "application/octet-stream".parse().unwrap(),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, size.to_string().parse().unwrap());
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::{Backend, EngineFuture, EngineState},
        scheduler::{self, Action},
        storage::Store,
    };
    use axum::{body::to_bytes, http::Request};
    use serde_json::{Value, json};
    use std::path::Path;
    use tokio::sync::{Semaphore, watch};
    use tower::ServiceExt;
    struct NoInference;
    impl Backend for NoInference {
        fn generate<'a>(
            &'a mut self,
            _: &'a crate::generation::ImagePass,
        ) -> EngineFuture<'a, Vec<u8>> {
            Box::pin(async { anyhow::bail!("no inference in preview fixture") })
        }
        fn stop(&mut self) -> EngineFuture<'_, ()> {
            Box::pin(async { Ok(()) })
        }
    }
    fn metadata() -> crate::h3_protocol::preview::Metadata {
        crate::h3_protocol::preview::Metadata {
            sequence: 1,
            step: 0,
            width: 1,
            height: 1,
            frames: 2,
            frame_duration_ms: 500,
            source_duration_seconds: 1.,
        }
    }
    async fn request(
        router: &Router,
        method: &str,
        uri: &str,
        session: Option<&str>,
        body: Option<Value>,
    ) -> Response {
        let mut request = Request::builder().method(method).uri(uri);
        if let Some(session) = session {
            request = request.header("X-Imagine-Session", session);
        }
        let body = if let Some(value) = body {
            request = request.header(header::CONTENT_TYPE, "application/json");
            Body::from(value.to_string())
        } else {
            Body::empty()
        };
        router
            .clone()
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap()
    }
    #[tokio::test]
    async fn binary_snapshot_remains_atomic_after_hub_release() {
        let hub = Hub::default();
        hub.begin("own-fixture");
        assert!(hub.publish("own-fixture", metadata(), vec![1, 2, 3, 4, 5, 6]));
        let clip = hub.snapshot("own-fixture", 0).unwrap();
        let weak = Arc::downgrade(&clip);
        let response = binary_response(clip).unwrap();
        hub.end("own-fixture");
        assert!(weak.upgrade().is_some());
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        let length = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        let decoded: crate::h3_protocol::preview::Metadata =
            serde_json::from_slice(&bytes[4..4 + length]).unwrap();
        assert_eq!(decoded, metadata());
        assert_eq!(&bytes[4 + length..], [1, 2, 3, 4, 5, 6]);
        drop(bytes);
        assert!(weak.upgrade().is_none());
    }
    #[tokio::test]
    async fn preview_http_requires_session_and_owned_job_without_persisting_pixels() {
        let root = tempfile::tempdir_in(crate::task_fixtures::env_root("TMPDIR")).unwrap();
        let store = Store::open(root.path()).await.unwrap();
        // These are NEW synthetic RAM database rows, never an existing runtime DB.
        for (id, owner) in [("own-preview", "local"), ("own-other-owner", "other")] {
            sqlx::query("INSERT INTO jobs(id,owner,status,request,request_hash,created) VALUES (?,?,'running','{}','fixture',0)")
                .bind(id).bind(owner).execute(&store.db).await.unwrap();
        }
        let (control, scheduler) = scheduler::start(store.clone(), Box::new(NoInference))
            .await
            .unwrap();
        let hub = Arc::new(Hub::default());
        let (_, engine) = watch::channel(EngineState {
            sampling_preview: hub.clone(),
            ..Default::default()
        });
        let app = App {
            video: control.video.clone(),
            enhancement: crate::enhancement::Service::open(root.path().join("enhancement.json"))
                .unwrap(),
            library: Default::default(),
            profiles: Arc::new(
                crate::profiles::Catalog::load(
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles"),
                )
                .unwrap(),
            ),
            store: store.clone(),
            control: control.clone(),
            engine,
            uploads: Arc::new(Semaphore::new(2)),
        };
        let router = routes().with_state(app);
        let session = store.session.id.as_str();
        let response = request(&router, "GET", "/v1/runtime/sampling-preview", None, None).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        assert_eq!(
            request(
                &router,
                "GET",
                "/v1/runtime/sampling-preview",
                Some("wrong"),
                None
            )
            .await
            .status(),
            StatusCode::CONFLICT
        );
        let bad = json!({"interval_seconds":0});
        assert_eq!(
            request(
                &router,
                "POST",
                "/v1/runtime/sampling-preview",
                Some(session),
                Some(bad)
            )
            .await
            .status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
        let configured = request(
            &router,
            "POST",
            "/v1/runtime/sampling-preview",
            Some(session),
            Some(json!({"interval_seconds":2,"max_resolution":999,"frames":999})),
        )
        .await;
        let configured: Value =
            serde_json::from_slice(&to_bytes(configured.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(configured["settings"]["max_resolution"], 999);
        assert_eq!(configured["effective"]["max_resolution"], 768);
        assert_eq!(configured["effective"]["frames"], 48);
        hub.begin("own-preview");
        assert!(hub.publish("own-preview", metadata(), vec![9; 6]));
        hub.warn("own-preview", "fast_preview_failed");
        let status = request(
            &router,
            "GET",
            "/v1/runtime/sampling-preview",
            Some(session),
            None,
        )
        .await;
        let status: Value =
            serde_json::from_slice(&to_bytes(status.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(status["active_job"], "own-preview");
        assert_eq!(status["metadata"]["sequence"], 1);
        assert_eq!(status["warning"], "fast_preview_failed");
        assert!(status.get("bytes").is_none());
        for id in ["missing-job", "own-other-owner"] {
            assert_eq!(
                request(
                    &router,
                    "GET",
                    &format!("/v1/jobs/{id}/sampling-preview"),
                    Some(session),
                    None
                )
                .await
                .status(),
                StatusCode::NOT_FOUND
            );
        }
        let response = request(
            &router,
            "GET",
            "/v1/jobs/own-preview/sampling-preview?after=0",
            Some(session),
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        let length = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert_eq!(&bytes[4 + length..], [9; 6]);
        assert_eq!(
            request(
                &router,
                "GET",
                "/v1/jobs/own-preview/sampling-preview?after=1",
                Some(session),
                None
            )
            .await
            .status(),
            StatusCode::NO_CONTENT
        );
        sqlx::query("UPDATE jobs SET status='succeeded' WHERE id='own-preview'")
            .execute(&store.db)
            .await
            .unwrap();
        assert_eq!(
            request(
                &router,
                "GET",
                "/v1/jobs/own-preview/sampling-preview?after=0",
                Some(session),
                None
            )
            .await
            .status(),
            StatusCode::NO_CONTENT
        );
        assert!(hub.status().active_job.is_none());
        hub.end("own-preview");
        assert_eq!(
            request(
                &router,
                "GET",
                "/v1/jobs/own-preview/sampling-preview",
                Some(session),
                None
            )
            .await
            .status(),
            StatusCode::NO_CONTENT
        );
        drop(router);
        control.call(Action::Shutdown).await.unwrap();
        scheduler.await.unwrap().unwrap();
        store.close().await.unwrap();
    }
}
