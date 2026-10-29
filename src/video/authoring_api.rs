use super::*;
impl From<Fault> for Error {
    fn from(fault: Fault) -> Self {
        Error::new(
            StatusCode::from_u16(fault.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            fault.code,
        )
    }
}
use super::super::authoring::{budget, drafts, engine, grounding::Grounding, steps};
use crate::scheduler::Fault;
use imagine_video_contract::authoring::{
    Draft, Measurement, Operation, Production, Request as AuthorRequest, Review, Submission,
};
/// Agent guidance beside the prompt profiles; never sent to the local provider.
const AGENT_GUIDANCE: &str = "video/agent-guidance.txt";
/// The error envelope, with where in a submission a problem was found.
struct Located {
    error: Error,
    at: Option<String>,
    detail: Option<String>,
}
impl IntoResponse for Located {
    fn into_response(self) -> Response {
        let mut body = json!({"error":{"code":self.error.code}});
        if let Some(at) = self.at {
            body["error"]["at"] = json!(at);
        }
        if let Some(detail) = self.detail {
            body["error"]["detail"] = json!(detail);
        }
        (self.error.status, Json(body)).into_response()
    }
}
impl From<Error> for Located {
    fn from(error: Error) -> Self {
        Self {
            error,
            at: None,
            detail: None,
        }
    }
}
impl From<Fault> for Located {
    fn from(fault: Fault) -> Self {
        Error::from(fault).into()
    }
}
impl From<StorageError> for Located {
    fn from(error: StorageError) -> Self {
        Error::from(error).into()
    }
}
impl From<Issue> for Located {
    fn from(issue: Issue) -> Self {
        Error::from(issue).into()
    }
}
impl From<steps::Rejection> for Located {
    fn from(r: steps::Rejection) -> Self {
        Self {
            error: Fault::new(r.status, r.code).into(),
            at: r.at,
            detail: r.detail,
        }
    }
}
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/authoring/profiles", get(profiles))
        .route("/v1/video/authoring/profiles/{id}", get(profile))
        .route("/v1/video/projects/{id}/authoring", post(start).get(list))
        .route("/v1/video/projects/{id}/authoring/measure", post(measure))
        .route("/v1/video/authoring/{id}", get(read))
        .route("/v1/video/authoring/{id}/bundle", get(bundle))
        .route("/v1/video/authoring/{id}/review", post(review))
        .route("/v1/video/authoring/{id}/expand", post(expand))
        .route("/v1/video/authoring/{id}/cancel", post(cancel))
        .route("/v1/video/authoring/{id}/retry", post(retry))
        .route("/v1/video/authoring/{id}/apply", post(apply))
        .route("/v1/video/authoring/{id}/submission", post(submission))
}
fn catalog() -> Result<super::super::authoring::profiles::Catalog, Fault> {
    super::super::authoring::profiles::Catalog::load(&crate::config_files::path(
        "video/prompt-profiles",
    ))
    .map_err(|c| Fault::new(422, c))
}
async fn profiles() -> Result<Json<Value>, Fault> {
    Ok(Json(json!(catalog()?)))
}
/// One profile exactly as requested, never a fallback, with the agent
/// guidance and planning limits.
async fn profile(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Fault> {
    let catalog = catalog()?;
    let profile = catalog
        .profiles
        .iter()
        .find(|p| p.id == id)
        .ok_or(Fault::new(404, "profile_not_found"))?;
    Ok(Json(
        json!({"profile":profile,"agent_guidance":crate::config_files::required(AGENT_GUIDANCE),"limits":limits(&app).await}),
    ))
}
/// Limits with the current preset catalog.
async fn limits(app: &App) -> Value {
    let service = app.video.lock().await;
    crate::video::presets::limits(
        service.video_presets.as_deref(),
        service.h3_model_root.as_deref(),
    )
}
async fn registry(app: &App) -> std::sync::Arc<super::super::authoring::drafts::Service> {
    app.video.lock().await.authoring.clone()
}
async fn start(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<AuthorRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), Located> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    if let Some(expected) = &request.budget_fingerprint {
        let mut unmeasured = request.clone();
        unmeasured.budget_fingerprint = None;
        if &measured(&app, &id, unmeasured).await?.fingerprint != expected {
            return Err(Fault::new(409, "authoring_budget_changed").into());
        }
    }
    let external = request.provider == steps::EXTERNAL;
    let provider = if external {
        crate::enhancement::ProviderSnapshot {
            profile: Default::default(),
        }
    } else {
        app.enhancement.snapshot(&request.provider_profile_id)?
    };
    let (base, authoring, grounding, identities) = {
        let service = app.video.lock().await;
        let session = service.projects.session(&ProjectId(id.clone()))?;
        let base = service.projects.read(&ProjectId(id)).map_err(Error::from)?;
        steps::check(&request, &base)?;
        let grounding = Grounding::capture(session, &request).map_err(media_error)?;
        let identities = if external {
            Grounding::identities(session, &request).map_err(media_error)?
        } else {
            vec![]
        };
        (base, service.authoring.clone(), grounding, identities)
    };
    let draft = authoring.start(base, request, provider, app.enhancement.clone(), grounding)?;
    let mut value = json!(draft);
    if external {
        value["bundle"] = authoring.bundle(&draft.id, &identities)?;
        value["bundle"]["limits"] = limits(&app).await;
    }
    Ok((StatusCode::ACCEPTED, Json(value)))
}
/// What an outline start of this request would send, measured; nothing starts.
async fn measure(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<AuthorRequest>, JsonRejection>,
) -> Result<Json<Value>, Located> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    if request.budget_fingerprint.is_some() {
        return Err(Fault::new(422, "invalid_authoring_request").into());
    }
    Ok(Json(json!(measured(&app, &id, request).await?)))
}
/// The measurement of a provider outline whose story images all have known
/// inventory notes, built exactly as its start builds the messages.
async fn measured(app: &App, id: &str, request: AuthorRequest) -> Result<Measurement, Located> {
    if request.operation != Operation::Outline || request.provider == steps::EXTERNAL {
        return Err(Fault::new(422, "invalid_authoring_request").into());
    }
    if request.inventory.len() != request.images.len() {
        return Err(steps::Rejection::new("authoring_inventory_mismatch", "inventory").into());
    }
    let provider = app.enhancement.snapshot(&request.provider_profile_id)?;
    let (draft, profile, base) = {
        let service = app.video.lock().await;
        let project = ProjectId(id.into());
        let session = service.projects.session(&project)?;
        let base = service.projects.read(&project).map_err(Error::from)?;
        steps::check(&request, &base)?;
        // The measurement needs the cast, not the images' pixels.
        let mut cast = request.clone();
        cast.images.clear();
        cast.inventory.clear();
        let grounding = Grounding::capture(session, &cast).map_err(media_error)?;
        let (draft, profile) = drafts::prepare(&base, request, &provider, &grounding)?;
        (draft, profile, base)
    };
    let basis = budget::basis(&draft, &base, &profile).map_err(|c| Fault::new(422, c))?;
    let window = app
        .enhancement
        .context_window(&provider, &draft.request.provider)
        .await
        .map_err(|c| {
            Fault::new(
                if c == "authoring_context_window_unknown" {
                    422
                } else {
                    502
                },
                c,
            )
        })?;
    Ok(budget::measure(&basis, &draft, &provider, &profile, window)
        .map_err(|c| Fault::new(422, c))?)
}
/// The project's Drafts in this session, so an agent can recover one after a timeout.
async fn list(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let service = app.video.lock().await;
    let id = ProjectId(id);
    service.projects.session(&id)?;
    Ok(Json(
        json!({"project_id":id,"drafts":service.authoring.list(&id)}),
    ))
}
/// An external Draft's context bundle again, built as at start.
async fn bundle(State(app): State<App>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    let service = app.video.lock().await;
    let draft = service.authoring.read(&id)?;
    if draft.request.provider != steps::EXTERNAL {
        return Err(Fault::new(409, "authoring_not_external").into());
    }
    let session = service.projects.session(&draft.project_id)?;
    let identities = Grounding::identities(session, &draft.request).map_err(media_error)?;
    let mut bundle = service.authoring.bundle(&id, &identities)?;
    bundle["limits"] = crate::video::presets::limits(
        service.video_presets.as_deref(),
        service.h3_model_root.as_deref(),
    );
    Ok(Json(
        json!({"draft_id":draft.id,"phase":draft.phase,"bundle":bundle}),
    ))
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
) -> Result<Json<Value>, Located> {
    let Json(review) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    let service = app.video.lock().await;
    if let Some(opening) = &review.opening_appearance {
        steps::review_opening(&service.authoring.read(&id)?, opening)?;
    }
    Ok(Json(json!(service.authoring.review(&id, review)?)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expand {
    schema_version: u32,
    start_index: usize,
    /// Regenerates only that Scene.
    #[serde(default)]
    only: bool,
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
        request.only,
        app.enhancement.clone()
    )?)))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Apply {
    schema_version: u32,
    expected_project_revision: u64,
    #[serde(default)]
    production: Option<Production>,
    #[serde(default)]
    dry_run: bool,
}
async fn apply(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Apply>, JsonRejection>,
) -> Result<Json<Value>, Located> {
    let Json(request) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    if request.schema_version != 1 {
        return Err(Fault::new(422, "unsupported_schema").into());
    }
    let mut service = app.video.lock().await;
    let mut draft = service.authoring.read(&id)?;
    let current = service.projects.read(&draft.project_id)?;
    if request.expected_project_revision != current.project_revision {
        return Err(Fault::new(409, "authoring_revision_conflict").into());
    }
    if request.production.is_some() {
        draft.production = request.production;
    }
    finish(&mut service, &id, &current, draft, request.dry_run)
}
/// An external agent's whole Story in one call, parsed by the provider
/// path's steps; all or nothing.
async fn submission(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Submission>, JsonRejection>,
) -> Result<Json<Value>, Located> {
    let Json(submission) = body.map_err(|_| Fault::new(422, "invalid_request"))?;
    if submission.schema_version != 1 {
        return Err(Fault::new(422, "unsupported_schema").into());
    }
    let mut service = app.video.lock().await;
    let draft = service.authoring.read(&id)?;
    let current = service.projects.read(&draft.project_id)?;
    if submission.expected_project_revision != current.project_revision
        || submission.expected_project_revision != draft.request.expected_project_revision
    {
        return Err(Fault::new(409, "authoring_revision_conflict").into());
    }
    let draft = service.authoring.submit(&id, &submission)?;
    finish(&mut service, &id, &current, draft, submission.dry_run)
}
/// Preflight codes for a continuity source that has no accepted take yet.
const PENDING: [&str; 2] = [
    "continuity_source_not_accepted",
    "continuity_source_missing",
];
/// Validates the Draft's apply plan; a dry run checks it on a copy of the
/// project and reports per-Scene preflight, a commit publishes it. The stored
/// Draft changes only when this succeeds.
fn finish(
    service: &mut super::super::service::Service,
    id: &str,
    current: &VideoProject,
    draft: Draft,
    dry_run: bool,
) -> Result<Json<Value>, Located> {
    let presets = crate::video::presets::Catalog::load(service.video_presets.as_deref());
    engine::production(current, &draft, &presets).map_err(steps::Rejection::from)?;
    let plan = engine::apply(current, &draft, &presets).map_err(|code| match code {
        "authoring_revision_conflict" => Fault::new(409, code).into(),
        // The Story's timing and image bindings come from the outline.
        _ if code.starts_with("story_") || code == "invalid_story_duration" => {
            steps::Rejection::new(code, "outline").into()
        }
        _ => Located::from(Fault::new(422, code)),
    })?;
    let added = plan
        .commands
        .iter()
        .filter_map(|c| match c {
            super::super::editing::Edit::AddScene { scene } => Some(scene.as_ref().clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    // A new Scene the project rejects is named by its expansion.
    let mut checked = current.clone();
    if let Err(issue) = super::super::editing::apply(&mut checked, plan.clone()) {
        let ordinal = issue
            .scene_id
            .as_ref()
            .and_then(|id| added.iter().position(|s| &s.id == id));
        return Err(match ordinal {
            Some(i) => Located {
                error: Error::from(Issue::new(&issue.code, &issue.field)),
                at: Some(format!("expansions[{i}]")),
                detail: Some(issue.field).filter(|f| !f.is_empty()),
            },
            None => issue.into(),
        });
    }
    if dry_run {
        let candidate = service
            .projects
            .session(&current.id)?
            .authoring_candidate(plan, draft.clone())?
            .project;
        let mut warnings = vec![];
        let cast = draft
            .selected_cast
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| c["reference_tag"].as_str())
            .map(|tag| tag.trim_start_matches('@').to_ascii_lowercase())
            .collect::<Vec<_>>();
        let scenes = added
            .iter()
            .enumerate()
            .map(|(i, added)| {
                let scene = candidate.scene(&added.id)?;
                let model = &candidate.effective_model(scene).variant;
                // I2V has no subject references; the compiler expands cast tags to descriptions.
                if model == "i2v" {
                    let text = std::iter::once(scene.description.as_str())
                        .chain(scene.sections.iter().map(|s| s.prompt.as_str()))
                        .chain(
                            candidate
                                .creative
                                .scene_prompts
                                .get(&scene.id)
                                .map(|p| p.summary.as_str()),
                        )
                        .collect::<Vec<_>>()
                        .join("\n");
                    let named = imagine_video_contract::references::aliases(&text);
                    let named = cast.iter().filter(|a| named.contains(*a)).collect::<Vec<_>>();
                    if !named.is_empty() {
                        warnings.push(json!({"ordinal":i+1,"code":"i2v_cast_identity_text_only","cast":named}));
                    }
                }
                let mut blockers = super::super::execution::preflight(&candidate, &scene.id);
                // A preset's model or LoRA files must exist where the worker looks.
                if let Some(root) = &service.h3_model_root
                    && !super::super::model_files::missing(&candidate, scene, root).is_empty()
                {
                    blockers.push("h3_model_file_missing".into());
                }
                // A continuous sequence planned before rendering waits for its
                // source take; admission checks it.
                if let Some(continuity) = &scene.continuity
                    && candidate
                        .scene(&continuity.source_scene)
                        .is_ok_and(|s| s.accepted_take.is_none())
                    && blockers.iter().any(|b| PENDING.contains(&b.as_str()))
                {
                    blockers.retain(|b| !PENDING.contains(&b.as_str()));
                    warnings.push(json!({"ordinal":i+1,"code":"continuity_source_pending","source_scene":continuity.source_scene}));
                }
                let boundary = draft
                    .scenes
                    .get(i)
                    .and_then(Option::as_ref)
                    .map(|s| s.boundary_from_previous.as_str());
                let preset = draft
                    .production
                    .as_ref()
                    .and_then(|p| p.scenes.get(i))
                    .map(|choice| {
                        choice.preset.clone().unwrap_or_else(|| {
                            crate::video::presets::default_key(model).to_owned()
                        })
                    });
                Ok(json!({"ordinal":i+1,"preset":preset,"model":model,"boundary":boundary,"continuity":scene.continuity,"planned_seconds":scene.duration.num as f64/scene.duration.den as f64,"blockers":blockers}))
            })
            .collect::<Result<Vec<_>, Issue>>()?;
        let valid = scenes.iter().all(|s| s["blockers"] == json!([]));
        service.authoring.store(id, draft.clone());
        return Ok(Json(
            json!({"valid":valid,"draft_id":draft.id,"project_revision":current.project_revision,"scenes":scenes,"warnings":warnings}),
        ));
    }
    service
        .projects
        .session_mut(&current.id)?
        .apply_authoring(plan, draft.clone())?;
    service.authoring.store(id, draft);
    service.authoring.applied(id);
    let mut value = json!(service.snapshot(&current.id)?);
    value["scene_map"] = json!(
        added
            .iter()
            .enumerate()
            .map(|(i, s)| json!({"ordinal":i+1,"scene_id":s.id,"section_ids":s.sections.iter().map(|s|&s.id).collect::<Vec<_>>()}))
            .collect::<Vec<_>>()
    );
    Ok(Json(value))
}
