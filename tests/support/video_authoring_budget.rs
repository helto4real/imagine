//! Inventory-only Drafts, known inventory notes and the outline budget,
//! measured exactly as the outline start sends its messages.
use super::authoring_messages::{grounded, provider, settle, story_request, user_text};
use super::authoring_tests::fixture;
use super::*;
use std::sync::atomic::Ordering;

async fn post(h: &Harness, path: &str, body: Value) -> (u16, Value) {
    call(h, "POST", path, Some(body)).await
}
/// A grounded outline request with its cast and four story images.
async fn outline(h: &Harness, project: &str, images: &Value) -> Value {
    let mut request = story_request(h, project, "koboldcpp", "normal").await;
    request["cast"] = json!(["keeper_id", "visitor_id"]);
    request["images"] = images.clone();
    request
}
/// The inventory notes an inventory Draft writes for `request`.
async fn inventory(h: &Harness, project: &str, request: &Value) -> Value {
    let mut start = request.clone();
    start["operation"] = json!("inventory");
    let (code, d) = post(h, &format!("/v1/video/projects/{project}/authoring"), start).await;
    assert_eq!(code, 202, "{d}");
    let draft = settle(h, d["id"].as_str().unwrap(), "review").await;
    draft["inventory"].clone()
}
#[tokio::test]
async fn an_inventory_draft_writes_only_notes_and_an_outline_reuses_them() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "inventory-reuse").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, images, _) = grounded(&h).await;
    let request = outline(&h, &project, &images).await;
    let mut start = request.clone();
    start["operation"] = json!("inventory");
    let (code, d) = post(
        &h,
        &format!("/v1/video/projects/{project}/authoring"),
        start,
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap().to_owned();
    let draft = settle(&h, &id, "review").await;
    let notes = draft["inventory"].clone();
    assert_eq!(
        notes
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["own_setting", "own_start", "own_end", "own_moment"]
    );
    assert!(draft["outline"].is_null());
    // Four image analyses and no outline.
    let sent = std::mem::take(&mut *state.requests.lock().unwrap());
    assert_eq!(sent.len(), 4);
    assert!(
        sent.iter()
            .all(|b| user_text(b).contains("Stable image ID:"))
    );
    // An inventory changes no project and has nothing to review.
    let (code, d) = post(
        &h,
        &format!("/v1/video/authoring/{id}/review"),
        json!({"schema_version":1,"outline":null,"scenes":null,"text":"Own text"}),
    )
    .await;
    assert_eq!(
        (code, d["error"]["code"].clone()),
        (409, json!("authoring_not_editable"))
    );
    let (code, d) = post(
        &h,
        &format!("/v1/video/authoring/{id}/apply"),
        json!({"schema_version":1,"expected_project_revision":request["expected_project_revision"]}),
    )
    .await;
    assert_eq!(code, 422, "{d}");
    assert_eq!(d["error"]["code"], "authoring_not_applicable");
    // The outline with those notes analyses no image and carries every note.
    let mut start = request.clone();
    start["inventory"] = notes.clone();
    let (code, d) = post(
        &h,
        &format!("/v1/video/projects/{project}/authoring"),
        start,
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let draft = settle(&h, d["id"].as_str().unwrap(), "outline").await;
    assert_eq!(draft["inventory"], notes);
    let sent = std::mem::take(&mut *state.requests.lock().unwrap());
    assert_eq!(sent.len(), 1, "only the outline");
    let user = user_text(&sent[0]);
    for note in notes.as_array().unwrap() {
        let escaped = serde_json::to_string(note["note"].as_str().unwrap()).unwrap();
        assert!(user.contains(escaped.trim_matches('"')), "{user}");
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn the_measurement_is_what_the_outline_start_sends_and_guards_its_basis() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "outline-budget").await;
    let (url, worker, state) = fixture::spawn().await;
    state.context_tokens.store(65536, Ordering::SeqCst);
    provider(&h, &url).await;
    let (project, images, _) = grounded(&h).await;
    let mut request = outline(&h, &project, &images).await;
    request["inventory"] = inventory(&h, &project, &request).await;
    state.requests.lock().unwrap().clear();
    let measure = format!("/v1/video/projects/{project}/authoring/measure");
    let (code, measured) = post(&h, &measure, request.clone()).await;
    assert_eq!(code, 200, "{measured}");
    assert_eq!(measured["schema_version"], 1);
    assert_eq!(measured["operation"], "outline");
    assert_eq!(measured["message_count"], 2);
    assert_eq!(measured["context_tokens"], 65536);
    assert_eq!(measured["context_reserve_tokens"], 8);
    assert_eq!(measured["char_limit"], 100000);
    assert!(state.requests.lock().unwrap().is_empty(), "nothing is sent");
    // The same request measures the same.
    assert_eq!(post(&h, &measure, request.clone()).await.1, measured);
    let mut start = request.clone();
    start["budget_fingerprint"] = measured["fingerprint"].clone();
    let authoring = format!("/v1/video/projects/{project}/authoring");
    let (code, d) = post(&h, &authoring, start.clone()).await;
    assert_eq!(code, 202, "{d}");
    settle(&h, d["id"].as_str().unwrap(), "outline").await;
    let sent = std::mem::take(&mut *state.requests.lock().unwrap());
    assert_eq!(sent.len(), 1);
    let system = sent[0]["messages"][0]["content"].as_str().unwrap();
    let user = user_text(&sent[0]);
    assert_eq!(measured["text_bytes"], system.len() + user.len());
    assert_eq!(measured["chars"], user.chars().count());
    assert_eq!(measured["max_output_tokens"], sent[0]["max_tokens"]);
    // A changed message or context window is not the measured basis.
    let mut changed = start.clone();
    changed["instruction"] = json!("Own different instruction");
    let (code, d) = post(&h, &authoring, changed).await;
    assert_eq!(
        (code, d["error"]["code"].clone()),
        (409, json!("authoring_budget_changed"))
    );
    state.context_tokens.store(32768, Ordering::SeqCst);
    let (code, d) = post(&h, &authoring, start).await;
    assert_eq!(
        (code, d["error"]["code"].clone()),
        (409, json!("authoring_budget_changed"))
    );
    assert!(
        state.requests.lock().unwrap().is_empty(),
        "no changed start ran"
    );
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn an_unknown_window_missing_notes_or_misplaced_fields_fail_by_name() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "outline-budget-failures").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, images, _) = grounded(&h).await;
    let measure = format!("/v1/video/projects/{project}/authoring/measure");
    let authoring = format!("/v1/video/projects/{project}/authoring");
    let code_of =
        |(code, d): (u16, Value)| (code, d["error"]["code"].as_str().unwrap_or("").to_owned());
    // A brief without story images needs no notes; the model reports no window.
    let brief = story_request(&h, &project, "koboldcpp", "normal").await;
    assert_eq!(
        code_of(post(&h, &measure, brief.clone()).await),
        (422, "authoring_context_window_unknown".into())
    );
    let mut ollama = brief.clone();
    ollama["provider"] = json!("ollama");
    state.context_tokens.store(65536, Ordering::SeqCst);
    assert_eq!(
        code_of(post(&h, &measure, ollama).await),
        (422, "authoring_context_window_unknown".into())
    );
    assert_eq!(post(&h, &measure, brief.clone()).await.0, 200);
    // Story images need one known note each.
    let request = outline(&h, &project, &images).await;
    let (code, d) = post(&h, &measure, request.clone()).await;
    assert_eq!(
        (code, d["error"]["code"].clone(), d["error"]["at"].clone()),
        (
            422,
            json!("authoring_inventory_mismatch"),
            json!("inventory")
        )
    );
    let mut wrong = request.clone();
    wrong["inventory"] = json!([{"id":"own_setting","note":"a"},{"id":"own_setting","note":"b"},
        {"id":"own_end","note":"c"},{"id":"own_moment","note":"d"}]);
    assert_eq!(
        code_of(post(&h, &authoring, wrong).await),
        (422, "authoring_inventory_mismatch".into())
    );
    // Known notes and budgets belong to a provider outline.
    let mut measured = brief.clone();
    measured["budget_fingerprint"] = json!("sha256:own");
    assert_eq!(
        code_of(post(&h, &measure, measured).await),
        (422, "invalid_authoring_request".into())
    );
    let mut external = brief.clone();
    external["provider"] = json!("external");
    assert_eq!(
        code_of(post(&h, &measure, external).await),
        (422, "invalid_authoring_request".into())
    );
    let mut empty = brief.clone();
    empty["operation"] = json!("inventory");
    assert_eq!(
        code_of(post(&h, &authoring, empty).await),
        (422, "invalid_authoring_request".into())
    );
    // The limits publish the character limit.
    let (code, profile) = call(&h, "GET", "/v1/video/authoring/profiles/normal", None).await;
    assert_eq!(code, 200, "{profile}");
    assert_eq!(profile["limits"]["char_limit"], 100000);
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
