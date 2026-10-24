//! Agent-selected project folders on an own synthetic encrypted volume only.
use super::storage_links::{private_volume, setup_private};
use super::*;
use std::path::PathBuf;

fn location(path: &Path, create: bool) -> Value {
    json!({"schema_version":1,"path":path,"create":create})
}
async fn located(h: &Harness, location_id: &Value, project: Value) -> (u16, Value) {
    call(
        h,
        "POST",
        "/v1/video/projects/at-location",
        Some(json!({"schema_version":1,"location_id":location_id,"project":project})),
    )
    .await
}
fn prompt(revision: u64, text: &str) -> Value {
    json!({"schema_version":1,"expected_project_revision":revision,
        "commands":[{"operation":"set_global_prompt","prompt":text}]})
}

#[tokio::test]
async fn private_project_at_an_approved_location_autosaves_and_reopens_from_its_folder() {
    let directory = tempfile::tempdir().unwrap();
    let (private, _) = private_volume(directory.path());
    let inputs = directory.path().join("own-volume/own-inputs");
    std::fs::create_dir(&inputs).unwrap();
    let (private, ignored) = private.with_project_roots(std::slice::from_ref(&inputs));
    assert!(ignored.is_empty());
    let h = setup_private(&private).await;
    let outside = directory.path().join("own-outside");
    for (path, code) in [
        (
            inputs.join("own-missing/own-project"),
            "location_parent_missing",
        ),
        (outside.clone(), "invalid_storage_root"),
        (
            private.layout().root().join("projects/own-chosen"),
            "invalid_project_location",
        ),
        (PathBuf::from("own-relative"), "invalid_project_location"),
    ] {
        let (status, error) = call(
            &h,
            "POST",
            "/v1/video/locations",
            Some(location(&path, true)),
        )
        .await;
        assert_eq!(status, 422, "{error}");
        assert_eq!(error["error"]["code"], code, "{path:?}");
        assert!(!path.is_absolute() || !path.exists());
    }
    assert!(!inputs.join("own-missing").exists());
    let folder = inputs.join("own-agent-project");
    let (status, _) = call(
        &h,
        "POST",
        "/v1/video/locations",
        Some(location(&folder, false)),
    )
    .await;
    assert_eq!(status, 422, "Selecting without create never creates");
    assert!(!folder.exists());
    let (status, selected) = call(
        &h,
        "POST",
        "/v1/video/locations",
        Some(location(&folder, true)),
    )
    .await;
    assert_eq!(status, 201, "{selected}");
    assert!(folder.is_dir());
    let (status, created) = located(&h, &selected["location_id"], Value::Null).await;
    assert_eq!(status, 201, "{created}");
    assert_eq!(created["storage_layout"], 1);
    assert_eq!(created["persistence"]["state"], "saved");
    assert_eq!(created["persistence"]["location"], json!(folder));
    assert_eq!(
        created["project"]["delivery"],
        json!(imagine_server::video_defaults::new_project().delivery)
    );
    let id = created["project"]["id"].as_str().unwrap().to_owned();
    assert!(folder.join("project.json").is_file());
    assert!(!private.layout().project(&id).unwrap().root.exists());
    let base = format!("/v1/video/projects/{id}");
    let (status, edited) = call(
        &h,
        "POST",
        &format!("{base}/edits"),
        Some(prompt(0, "own located text")),
    )
    .await;
    assert_eq!(status, 200, "{edited}");
    assert_eq!(edited["persistence"]["saved_revision"], 1);
    assert!(
        std::fs::read_to_string(folder.join("project.json"))
            .unwrap()
            .contains("own located text")
    );
    let destination = h.store.output_destination(Some(&id), true).unwrap();
    let proof = destination.proof().unwrap();
    assert_eq!(proof.config.root, folder.join("outputs/videos"));
    h.store.destination_for_proof(&json!(proof)).unwrap();
    let (status, saved) = call(
        &h,
        "POST",
        &format!("{base}/save"),
        Some(json!({"schema_version":1,"expected_project_revision":1})),
    )
    .await;
    assert_eq!(status, 200, "{saved}");

    let (status, again) = located(&h, &selected["location_id"], create()).await;
    assert_eq!(status, 200, "The project bound there: {again}");
    assert_eq!(again["project"]["id"], id.as_str());
    std::fs::create_dir(&outside).unwrap();
    let elsewhere = select(&h, &outside).await;
    let (status, error) = located(&h, &json!(elsewhere), create()).await;
    assert_eq!(status, 422, "{error}");
    assert_eq!(error["error"]["code"], "invalid_storage_root");
    assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);

    let (status, named) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    assert_eq!(status, 201, "{named}");
    let named_id = named["project"]["id"].as_str().unwrap().to_owned();
    let named_folder = private.layout().project(&named_id).unwrap().root;
    assert_eq!(named["persistence"]["location"], json!(named_folder));
    stop(h).await;

    let h = setup_private(&private).await;
    let reselected = select(&h, &folder).await;
    let (status, opened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":reselected})),
    )
    .await;
    assert_eq!(status, 200, "{opened}");
    assert_eq!(opened["project"]["id"], id.as_str());
    assert_eq!(opened["project"]["global_prompt"], "own located text");
    assert_eq!(opened["persistence"]["location"], json!(folder));
    assert!(opened["persistence"]["error_code"].is_null());
    let (status, edited) = call(
        &h,
        "POST",
        &format!("{base}/edits"),
        Some(prompt(1, "own reopened text")),
    )
    .await;
    assert_eq!(status, 200, "{edited}");
    assert_eq!(edited["persistence"]["saved_revision"], 2);
    assert_eq!(
        h.store
            .output_destination(Some(&id), true)
            .unwrap()
            .proof()
            .unwrap()
            .config
            .root,
        folder.join("outputs/videos")
    );
    let named_location = select(&h, &named_folder).await;
    let (status, reopened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":named_location})),
    )
    .await;
    assert_eq!(status, 200, "{reopened}");
    assert_eq!(reopened["project"]["id"], named_id.as_str());
    let (status, edited) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{named_id}/edits"),
        Some(prompt(0, "own named text")),
    )
    .await;
    assert_eq!(status, 200, "{edited}");
    assert_eq!(edited["persistence"]["saved_revision"], 1);
    stop(h).await;

    let unapproved = tempfile::tempdir().unwrap();
    let (unconfigured, _) = private_volume(unapproved.path());
    let moved = unapproved
        .path()
        .join("own-volume/own-inputs/own-agent-project");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&folder, &moved).unwrap();
    let h = setup_private(&unconfigured).await;
    let selected = select(&h, &moved).await;
    let (status, error) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":selected})),
    )
    .await;
    assert_eq!(status, 422, "{error}");
    assert_eq!(error["error"]["code"], "private_storage_location_required");
    stop(h).await;
}

#[tokio::test]
async fn project_at_location_without_private_storage_saves_into_the_selected_folder() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "own-located-session").await;
    let missing = root.path().join("own-missing/own-project");
    let (status, error) = call(
        &h,
        "POST",
        "/v1/video/locations",
        Some(location(&missing, true)),
    )
    .await;
    assert_eq!(status, 422, "{error}");
    assert_eq!(error["error"]["code"], "location_parent_missing");
    assert!(!root.path().join("own-missing").exists());
    let folder = root.path().join("own-located");
    let (status, selected) = call(
        &h,
        "POST",
        "/v1/video/locations",
        Some(location(&folder, true)),
    )
    .await;
    assert_eq!(status, 201, "{selected}");
    let (status, created) = located(&h, &selected["location_id"], create()).await;
    assert_eq!(status, 201, "{created}");
    assert!(created["storage_layout"].is_null());
    assert_eq!(created["persistence"]["state"], "saved");
    assert_eq!(created["persistence"]["location"], json!(folder));
    assert_eq!(created["project"]["delivery"]["width"], 1280);
    assert!(folder.join("project.json").is_file());
    let (status, error) = located(&h, &json!("location_unknown"), Value::Null).await;
    assert_eq!(status, 404, "{error}");
    assert_eq!(error["error"]["code"], "location_not_found");
    stop(h).await;
}

#[tokio::test]
async fn located_create_is_idempotent_per_folder_and_never_nests_or_reuses_filled_folders() {
    let directory = tempfile::tempdir().unwrap();
    let (private, _) = private_volume(directory.path());
    let inputs = directory.path().join("own-volume/own-inputs");
    std::fs::create_dir(&inputs).unwrap();
    let (private, _) = private.with_project_roots(std::slice::from_ref(&inputs));
    let h = setup_private(&private).await;
    let create_location = |path: PathBuf| {
        let h = &h;
        async move {
            call(
                h,
                "POST",
                "/v1/video/locations",
                Some(location(&path, true)),
            )
            .await
        }
    };
    let folder = inputs.join("own-first");
    let (_, selected) = create_location(folder.clone()).await;
    let (status, created) = located(&h, &selected["location_id"], Value::Null).await;
    assert_eq!(status, 201, "{created}");
    // A retried create (both calls again) returns the project bound there.
    let (status, again) = create_location(folder.clone()).await;
    assert_eq!(status, 201, "{again}");
    let (status, retried) = located(&h, &again["location_id"], create()).await;
    assert_eq!(status, 200, "{retried}");
    assert_eq!(retried["project"]["id"], created["project"]["id"]);
    assert_eq!(retried["project"]["project_revision"], 0);
    // Never inside another project's folder: an open one, or a saved project.json.
    for nested in [folder.join("own-nested"), folder.join("work/own-nested")] {
        let (status, error) = create_location(nested.clone()).await;
        assert_eq!(status, 422, "{error}");
        assert_eq!(error["error"]["code"], "project_location_nested");
        assert!(!nested.exists());
    }
    let closed = inputs.join("own-closed");
    std::fs::create_dir_all(closed.join("own-existing")).unwrap();
    std::fs::write(closed.join("project.json"), b"{}").unwrap();
    let (status, error) = create_location(closed.join("own-inner")).await;
    assert_eq!(status, 422, "{error}");
    assert_eq!(error["error"]["code"], "project_location_nested");
    let inside = select(&h, &closed.join("own-existing")).await;
    let (status, error) = located(&h, &json!(inside), Value::Null).await;
    assert_eq!(status, 422, "{error}");
    assert_eq!(error["error"]["code"], "project_location_nested");
    // Prepared classes from an interrupted save count as empty only when they are.
    let prepared = inputs.join("own-prepared");
    std::fs::create_dir_all(prepared.join("outputs/videos")).unwrap();
    let (status, created) = located(&h, &json!(select(&h, &prepared).await), Value::Null).await;
    assert_eq!(status, 201, "{created}");
    let filled = inputs.join("own-filled");
    std::fs::create_dir_all(filled.join("outputs/videos")).unwrap();
    std::fs::write(filled.join("outputs/videos/own.mp4"), b"OWN").unwrap();
    let (status, error) = located(&h, &json!(select(&h, &filled).await), Value::Null).await;
    assert_eq!(status, 422, "{error}");
    assert_eq!(error["error"]["code"], "storage_not_empty");
    assert!(!filled.join("project.json").exists());
    stop(h).await;
}

#[tokio::test]
async fn project_at_location_takes_a_canvas_preset_and_resolves_its_delivery_size() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "own-located-canvas").await;
    let mut ids = vec![];
    for name in ["own-draft", "own-both", "own-unknown"] {
        let (status, selected) = call(
            &h,
            "POST",
            "/v1/video/locations",
            Some(location(&root.path().join(name), true)),
        )
        .await;
        assert_eq!(status, 201, "{selected}");
        ids.push(selected["location_id"].clone());
    }
    let canvas = |quality: &str| json!({"ratio":"16:9","portrait":false,"quality":quality});
    let at = |id: &Value, extra: Value| {
        let mut body = json!({"schema_version":1,"location_id":id});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    let (status, created) = call(
        &h,
        "POST",
        "/v1/video/projects/at-location",
        Some(at(&ids[0], json!({"canvas":canvas("Draft")}))),
    )
    .await;
    assert_eq!(status, 201, "{created}");
    let shape = imagine_server::h3_protocol::Selection {
        ratio: "16:9".into(),
        portrait: false,
        quality: "Draft".into(),
        seconds: 5.,
    }
    .resolve()
    .unwrap();
    assert_eq!(created["project"]["delivery"]["width"], shape.width);
    assert_eq!(created["project"]["delivery"]["height"], shape.height);
    assert_eq!(
        created["project"]["generation"]["canvas"]["quality"],
        "Draft"
    );
    for (id, extra, code) in [
        (
            &ids[1],
            json!({"canvas":canvas("Draft"),"project":create()}),
            "invalid_request",
        ),
        (
            &ids[2],
            json!({"canvas":canvas("Ultra")}),
            "invalid_h3_quality",
        ),
    ] {
        let (status, error) = call(
            &h,
            "POST",
            "/v1/video/projects/at-location",
            Some(at(id, extra)),
        )
        .await;
        assert_eq!(status, 422, "{error}");
        assert_eq!(error["error"]["code"], code);
    }
    let (_, capabilities) = call(&h, "GET", "/v1/video/capabilities", None).await;
    assert_eq!(
        capabilities["canvas"]["qualities"],
        json!(imagine_server::h3_protocol::geometry::QUALITIES)
    );
    assert_eq!(capabilities["canvas"]["default"]["quality"], "Quick Draft");
    stop(h).await;
}
