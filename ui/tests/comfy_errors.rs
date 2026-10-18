//! Synthetic HTTP responses; no inference, model imports or desktop UI.
use axum::{
    Json, Router,
    http::StatusCode,
    routing::{get, post},
};
use imagine_ui::{
    client::{Client, Submission},
    settings::Connection,
};
use serde_json::json;

#[tokio::test]
async fn comfy_errors_survive_submission_and_job_polling_without_raw_details() {
    for code in [
        "comfy_startup_failed",
        "comfy_startup_timeout",
        "comfy_kitchen_unavailable",
        "comfy_protocol_error",
        "comfy_generation_failed",
        "comfy_invalid_output",
        "comfy_input_too_large",
        "unexpected_output_dimensions",
    ] {
        for rejected_submission in [false, true] {
            let router = Router::new().route("/v1/jobs", post(move || async move {
                if rejected_submission {
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":{"code":code,"details":"synthetic private sentinel"}})))
                } else {
                    (StatusCode::ACCEPTED, Json(json!({"id":"job_synthetic"})))
                }
            })).route("/v1/jobs/{id}", get(move || async move {
                Json(json!({"id":"job_synthetic","status":"failed","error":{"code":code,"details":"synthetic private sentinel"}}))
            }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let client = Client::connect(&Connection {
                url: format!("http://{}", listener.local_addr().unwrap()),
            })
            .unwrap();
            let server = tokio::spawn(async move {
                axum::serve(listener, router).await.unwrap();
            });
            let directory = tempfile::tempdir().unwrap();
            let result = client
                .generate(
                    Submission {
                        request: json!({"prompt":"synthetic"}),
                        folder: directory.path().into(),
                        key: "synthetic".into(),
                        job: None,
                        upscale_source: None,
                    },
                    |_| {},
                )
                .await;
            let failure = match result {
                Err(failure) => failure,
                Ok(_) => panic!("failed job must not produce output"),
            };
            assert!(
                failure.message.contains(code),
                "lost diagnostic code: {}",
                failure.message
            );
            assert!(!failure.message.contains("The server could not complete"));
            assert!(!failure.message.contains("synthetic private sentinel"));
            assert_eq!(failure.retryable, rejected_submission);
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
            server.abort();
            let _ = server.await;
        }
    }
}
