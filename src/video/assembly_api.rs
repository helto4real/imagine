use super::*;
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/projects/{id}/assembly", post(plan))
        .route("/v1/video/assemblies/{id}", get(read))
        .route("/v1/video/assemblies/{id}/jobs", post(submit))
}
async fn plan(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<super::super::assembly::Request>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    if request.project_id.0 != id {
        return Err(Error::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "project_id_mismatch",
        ));
    }
    let staging = staging(&app)?;
    let service = app.video.clone().lock_owned().await;
    let plan = tokio::task::spawn_blocking(move || {
        let mut service = service;
        let frozen = super::super::assembly::freeze(&service, &request, &staging)?;
        let plan = frozen.plan.clone();
        service.assemblies.insert(
            plan.id.clone(),
            super::super::assembly::Entry {
                frozen,
                progress: Default::default(),
                job: None,
                rendered: Default::default(),
            },
        );
        Ok::<_, StorageError>(plan)
    })
    .await
    .map_err(internal)??;
    Ok(Json(json!(plan)))
}
async fn read(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let service = app.video.lock().await;
    let entry = service
        .assemblies
        .get(&id)
        .ok_or_else(|| Error::new(StatusCode::NOT_FOUND, "assembly_plan_not_found"))?;
    Ok(Json(
        json!({"plan":entry.frozen.plan,"progress":entry.progress.read(),"job_id":entry.job,"rendered_result_available":entry.rendered.get().is_some()}),
    ))
}
async fn submit(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
) -> Result<(StatusCode, Json<Value>), Error> {
    let key = headers
        .get("Idempotency-Key")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Error::new(StatusCode::BAD_REQUEST, "idempotency_key_required"))?;
    let result = app
        .control
        .call(crate::scheduler::Action::SubmitAssembly {
            owner: "local".into(),
            plan_id: id,
            key: key.into(),
        })
        .await?;
    Ok((StatusCode::ACCEPTED, Json(result)))
}
