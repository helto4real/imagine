//! The operator's execution choices; a change applies to jobs started after it.
use super::{App, Fault};
use crate::execution_settings::Settings;
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::HeaderMap,
    routing::get,
};

pub(super) fn routes() -> Router<App> {
    Router::new().route("/v1/runtime/execution", get(read).post(write))
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
async fn read(State(app): State<App>, headers: HeaderMap) -> Result<Json<Settings>, Fault> {
    session(&app, &headers)?;
    Ok(Json(app.engine.borrow().execution.get()))
}
async fn write(
    State(app): State<App>,
    headers: HeaderMap,
    body: Result<Json<Settings>, JsonRejection>,
) -> Result<Json<Settings>, Fault> {
    session(&app, &headers)?;
    let Json(settings) = body.map_err(|_| Fault::new(422, "invalid_execution_settings"))?;
    let store = app.engine.borrow().execution.clone();
    store.set(settings).map_err(|code| Fault::new(500, code))?;
    Ok(Json(store.get()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::{Backend, EngineFuture, EngineState},
        execution_settings::Attention,
        scheduler::{self, Action},
        storage::Store,
    };
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
    };
    use serde_json::{Value, json};
    use std::sync::Arc;
    use tokio::sync::{Semaphore, watch};
    use tower::ServiceExt;
    struct NoInference;
    impl Backend for NoInference {
        fn generate<'a>(
            &'a mut self,
            _: &'a crate::generation::ImagePass,
        ) -> EngineFuture<'a, Vec<u8>> {
            Box::pin(async { anyhow::bail!("no inference in execution fixture") })
        }
        fn stop(&mut self) -> EngineFuture<'_, ()> {
            Box::pin(async { Ok(()) })
        }
    }
    async fn call(
        router: &Router,
        method: &str,
        session: Option<&str>,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut request = Request::builder()
            .method(method)
            .uri("/v1/runtime/execution");
        if let Some(session) = session {
            request = request.header("X-Imagine-Session", session);
        }
        let body = match body {
            Some(value) => {
                request = request.header(header::CONTENT_TYPE, "application/json");
                Body::from(value.to_string())
            }
            None => Body::empty(),
        };
        let response = router
            .clone()
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }
    #[tokio::test]
    async fn choices_need_the_session_default_to_auto_and_apply_to_the_engine() {
        let root = tempfile::tempdir_in(crate::task_fixtures::env_root("TMPDIR")).unwrap();
        let store = Store::open(root.path()).await.unwrap();
        let (control, scheduler) = scheduler::start(store.clone(), Box::new(NoInference))
            .await
            .unwrap();
        let (_, engine) = watch::channel(EngineState::default());
        let execution = engine.borrow().execution.clone();
        let app = App {
            video: control.video.clone(),
            enhancement: crate::enhancement::Service::open(root.path().join("enhancement.json"))
                .unwrap(),
            library: Default::default(),
            profiles: Arc::new(
                crate::profiles::Catalog::load(&crate::config_files::path("image/profiles"))
                    .unwrap(),
            ),
            store: store.clone(),
            control: control.clone(),
            engine,
            uploads: Arc::new(Semaphore::new(2)),
        };
        let router = routes().with_state(app);
        let session = store.session.id.as_str();
        assert_eq!(
            call(&router, "GET", None, None).await.0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            call(&router, "GET", Some("wrong"), None).await.0,
            StatusCode::CONFLICT
        );
        assert_eq!(
            call(&router, "GET", Some(session), None).await,
            (
                StatusCode::OK,
                json!({"attention":"auto","dynamic_vram":true})
            )
        );
        for bad in [
            json!({"attention":"sage"}),
            json!({"attention":"auto","extra":1}),
        ] {
            let (status, body) = call(&router, "POST", Some(session), Some(bad)).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(body["error"]["code"], "invalid_execution_settings");
        }
        assert_eq!(execution.get().attention, Attention::Auto);
        let chosen = json!({"attention":"pytorch_sdpa","dynamic_vram":false});
        assert_eq!(
            call(&router, "POST", Some(session), Some(chosen.clone())).await,
            (StatusCode::OK, chosen.clone())
        );
        assert_eq!(execution.get().attention, Attention::PytorchSdpa);
        assert!(!execution.get().dynamic_vram);
        assert_eq!(call(&router, "GET", Some(session), None).await.1, chosen);
        drop(router);
        control.call(Action::Shutdown).await.unwrap();
        scheduler.await.unwrap().unwrap();
        store.close().await.unwrap();
    }
}
