use super::*;
#[tokio::test]
async fn more_than_sixty_four_completed_proposals_and_applied_records_remain_usable() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "authoring-many").await;
    let (url, worker, _) = fixture::spawn().await;
    let id = configured(&h, &url).await;
    let base = expanded(&h, &id, "append").await;
    let scene = base["project"]["scenes"][0]["id"].clone();
    for _ in 0..65 {
        let mut body = request(&h, &id, "enhance_scene", "koboldcpp").await;
        body["scene_id"] = scene.clone();
        let revision = body["expected_project_revision"].clone();
        let (code, draft) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{id}/authoring"),
            Some(body),
        )
        .await;
        assert_eq!(code, 202, "{draft}");
        let draft_id = draft["id"].as_str().unwrap();
        let ready = wait(&h, draft_id, "review").await;
        assert!(ready["text"].is_string());
        let (code, saved) = call(
            &h,
            "POST",
            &format!("/v1/video/authoring/{draft_id}/apply"),
            Some(json!({"schema_version":1,"expected_project_revision":revision})),
        )
        .await;
        assert_eq!(code, 200, "{saved}");
    }
    let (_, current) = call(&h, "GET", &format!("/v1/video/projects/{id}"), None).await;
    assert_eq!(current["authoring"].as_array().unwrap().len(), 66);
    worker.abort();
    stop(h).await;
}
#[tokio::test]
async fn compiled_scene_source_dialogue_and_pending_media_are_explicit_without_inference() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "compile-source").await;
    let (_, project) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let id = project["project"]["id"].as_str().unwrap();
    let scene = json!({"id":"own_scene","start":{"num":0,"den":1},"duration":{"num":5,"den":1},"description":"@keeper enters.","model":"project_default","settings":[],"sections":[{"id":"own_section","start":{"num":0,"den":1},"duration":{"num":5,"den":1},"prompt":"@keeper whispers [Swedish] \"Hej, världen!\" audio: waves"}],"bindings":[],"continuity":null,"accepted_take":null});
    let reference = json!({"id":"keeper_id","name":"Keeper","alias":"keeper","kind":"character","description":"the keeper in a blue coat","enabled":true,"strength":1.,"retention":"","images":[],"voice":{"language":"Swedish","description":"a quiet voice.","assets":[]},"origin":null});
    let creative = json!({"references":[reference],"sections":{},"scene_prompts":{"own_scene":{"title":"Arrival","summary":"A greeting","ambient_sound":"Wind","non_diegetic_music":"Soft piano"}}});
    let(code,edited)=call(&h,"POST",&format!("/v1/video/projects/{id}/edits"),Some(json!({"schema_version":1,"expected_project_revision":0,"commands":[{"operation":"add_scene","scene":scene},{"operation":"set_creative","creative":creative}]}))).await;
    assert_eq!(code, 200, "{edited}");
    let (code, compiled) = call(
        &h,
        "GET",
        &format!("/v1/video/projects/{id}/scenes/own_scene/compiled"),
        None,
    )
    .await;
    assert_eq!(code, 200, "{compiled}");
    let text = compiled["prompt"].as_str().unwrap();
    assert!(
        text.contains("<Subject 1> (S1) whispers in a quiet voice: <d>[Swedish] Hej, världen!</d>"),
        "{text}"
    );
    assert!(text.contains("Soft piano"));
    assert!(text.contains("[Shot 1] Starting shot:"));
    assert_eq!(compiled["runtime_issues"], json!([]));
    assert_eq!(compiled["project_revision"], 1);
    let (_, uploaded) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(png()),
        "image/png",
        true,
    )
    .await;
    let (_, attached) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{id}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":1,"asset_id":uploaded["id"]})),
    )
    .await;
    let asset = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap();
    let mut creative = creative;
    creative["references"][0]["images"] =
        json!([{"id":"own_image","asset":asset,"enabled":true,"preview":false}]);
    assert_eq!(call(&h,"POST",&format!("/v1/video/projects/{id}/edits"),Some(json!({"schema_version":1,"expected_project_revision":2,"commands":[{"operation":"set_creative","creative":creative}]}))).await.0,200);
    let (_, bound) = call(
        &h,
        "GET",
        &format!("/v1/video/projects/{id}/scenes/own_scene/compiled"),
        None,
    )
    .await;
    assert_eq!(
        bound["reference_manifest"]["subjects"][0]["source_images"][0]["asset_id"],
        asset.as_str()
    );
    assert_eq!(bound["runtime_issues"], json!([]));
    let staging = imagine_server::video::persistence::SessionStaging::select(root.path()).unwrap();
    let service = h.control.video.lock().await;
    let err = imagine_server::video::execution::freeze(
        &service,
        &imagine_server::video::execution::Admission {
            kind: imagine_server::video::execution::Kind::VideoScene,
            schema_version: 1,
            project_id: imagine_server::video::contract::ProjectId(id.into()),
            expected_project_revision: 3,
            scene_id: imagine_server::video::contract::SceneId("own_scene".into()),
            seed: 42,
        },
        &staging,
    )
    .err()
    .unwrap();
    assert_eq!(err.code, "scene_runtime_unsupported"); // Fixture has no executable model/settings; media itself is no longer blocked.
    // The editor's preflight reports the same admission blocker before submission.
    let project = service
        .projects
        .read(&imagine_server::video::contract::ProjectId(id.into()))
        .unwrap();
    let blockers = imagine_server::video::execution::preflight(
        &project,
        &imagine_server::video::contract::SceneId("own_scene".into()),
    );
    assert!(blockers.contains(&err.code), "{blockers:?}");
    drop(service);
    let (code, preflight) = call(
        &h,
        "GET",
        &format!("/v1/video/projects/{id}/scenes/own_scene/preflight"),
        None,
    )
    .await;
    assert_eq!(code, 200, "{preflight}");
    assert_eq!(preflight["project_revision"], 3);
    assert_eq!(preflight["blockers"], json!(blockers));
    let (_, capabilities) = call(&h, "GET", "/v1/video/capabilities", None).await;
    let variants = capabilities["models"].as_array().unwrap();
    assert_eq!(variants.len(), 2);
    let i2v = variants
        .iter()
        .find(|m| m["model"]["variant"] == "i2v")
        .unwrap();
    assert_eq!(
        i2v["capabilities"]["input_roles"],
        json!({"first_frame":["image"]})
    );
    stop(h).await;
}
#[tokio::test]
async fn enhancement_protects_restores_and_retains_source_on_missing_marker() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "reference-protection").await;
    let (url, worker, state) = fixture::spawn().await;
    let id = configured(&h, &url).await;
    let applied = expanded(&h, &id, "append").await;
    let scene = &applied["project"]["scenes"][0];
    let mut section = scene["sections"][0].clone();
    let source = "@keeper whispers [Swedish] \"Hej!\" then @keeper waves.";
    section["prompt"] = json!(source);
    assert_eq!(call(&h,"POST",&format!("/v1/video/projects/{id}/edits"),Some(json!({"schema_version":1,"expected_project_revision":applied["project"]["project_revision"],"commands":[{"operation":"update_section","scene_id":scene["id"],"section":section}]}))).await.0,200);
    let mut start = request(&h, &id, "enhance_section", "koboldcpp").await;
    start["scene_id"] = scene["id"].clone();
    start["section_id"] = section["id"].clone();
    state
        .drop_reference_once
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let (_, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{id}/authoring"),
        Some(start),
    )
    .await;
    let draft_id = d["id"].as_str().unwrap();
    let failed = wait(&h, draft_id, "failed").await;
    assert_eq!(failed["failure"], "authoring_reference_marker_missing");
    assert_eq!(failed["text"], source);
    state
        .introduce_reference_once
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/authoring/{draft_id}/retry"),
            Some(json!({}))
        )
        .await
        .0,
        200
    );
    let ready = wait(&h, draft_id, "review").await;
    assert!(ready["text"].as_str().unwrap().contains(source));
    assert!(!ready["text"].as_str().unwrap().contains("DIRECTOR_REF_"));
    assert_eq!(ready["warnings"].as_array().unwrap().len(), 1);
    let payload = state.requests.lock().unwrap().last().unwrap().clone();
    let text = payload["messages"][1]["content"].as_str().unwrap();
    assert!(text.contains("[[DIRECTOR_REF_0001]]"));
    assert!(
        !text
            .split("Source text:\n")
            .nth(1)
            .unwrap()
            .contains("@keeper")
    );
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn library_cast_copy_is_atomic_owned_and_independent_of_later_library_edits() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "cast-copy").await;
    let (_, project) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let project = project["project"]["id"].as_str().unwrap();
    let (_, uploaded) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(png()),
        "image/png",
        true,
    )
    .await;
    let folder = root.path().join("library");
    std::fs::create_dir(&folder).unwrap();
    let location = select(&h, &folder).await;
    let (code, library) = call(
        &h,
        "POST",
        "/v1/video/libraries",
        Some(json!({"schema_version":1,"location_id":location,"mode":"create"})),
    )
    .await;
    assert_eq!(code, 200, "{library}");
    let library = library["id"].as_str().unwrap();
    let character = json!({"name":"Own keeper","alias":"keeper","description":"Original owned description","tags":["own","fixture"],"visual_asset_ids":[uploaded["id"]],"voice":{"language":"Swedish","description":"calm","reference_asset_ids":[]}});
    let(code,entry)=call(&h,"POST",&format!("/v1/video/libraries/{library}/characters"),Some(json!({"schema_version":1,"expected_library_revision":0,"title":"Own keeper","replace":null,"character":character}))).await;
    assert_eq!(code, 200, "{entry}");
    let entry = entry["entry_id"].as_str().unwrap();
    let payload = json!({"schema_version":1,"expected_project_revision":0,"library_id":library,"entry_id":entry,"expected_entry_revision":1});
    sqlx::query("CREATE TRIGGER own_copy_fault BEFORE INSERT ON assets BEGIN SELECT RAISE(FAIL,'own copy fault'); END").execute(&h.store.db).await.unwrap();
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/cast/from-library"),
            Some(payload.clone())
        )
        .await
        .0,
        500
    );
    let (_, unchanged) = call(&h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    assert_eq!(unchanged["project"]["project_revision"], 0);
    assert!(unchanged["project"]["creative"].is_null());
    sqlx::query("DROP TRIGGER own_copy_fault")
        .execute(&h.store.db)
        .await
        .unwrap();
    let (code, copied) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/cast/from-library"),
        Some(payload),
    )
    .await;
    assert_eq!(code, 200, "{copied}");
    let reference = &copied["project"]["creative"]["references"][0];
    assert_eq!(reference["description"], "Original owned description");
    assert_eq!(reference["tags"], json!(["own", "fixture"]));
    assert_eq!(reference["voice"]["language"], "Swedish");
    assert_eq!(reference["origin"]["revision"], 1);
    let asset = reference["images"][0]["asset"].as_str().unwrap();
    let runtime = copied["runtime_assets"][asset].as_str().unwrap();
    assert_eq!(download(&h, runtime, None).await, (200, png()));
    let mut changed = character;
    changed["description"] = json!("Changed only in library");
    assert_eq!(call(&h,"POST",&format!("/v1/video/libraries/{library}/characters"),Some(json!({"schema_version":1,"expected_library_revision":1,"title":"Changed","replace":{"entry_id":entry,"expected_entry_revision":1},"character":changed}))).await.0,200);
    let (_, after) = call(&h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    assert_eq!(after["project"], copied["project"]);
    let target = root.path().join("owned-project");
    std::fs::create_dir(&target).unwrap();
    let location = select(&h, &target).await;
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/save"),
            Some(json!({"schema_version":1,"expected_project_revision":1,"location_id":location}))
        )
        .await
        .0,
        200
    );
    stop(h).await;
    std::fs::remove_dir_all(folder).unwrap();
    let reopened = imagine_server::video::persistence::ProjectSession::open(&target).unwrap();
    assert_eq!(reopened.document().project.creative.references.len(), 1);
    assert_eq!(
        reopened.document().project.creative.references[0].description,
        "Original owned description"
    );
}
#[tokio::test]
async fn cast_inventory_dialogue_retry_apply_and_reopen_keep_owned_bindings_and_exact_words() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "grounded-story").await;
    let (url, worker, state) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let (code, uploaded) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(png()),
        "image/png",
        true,
    )
    .await;
    assert_eq!(code, 201, "{uploaded}");
    let (code, attached) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":uploaded["id"]})),
    )
    .await;
    assert_eq!(code, 200, "{attached}");
    let asset = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    let reference = |id: &str, alias: &str| json!({"id":id,"name":alias,"alias":alias,"kind":"Character","description":format!("Own fictional {alias}"),"enabled":true,"strength":1.,"retention":"","images":[],"voice":{"language":"Swedish","description":"calm","assets":[]},"origin":null});
    let(code,edited)=call(&h,"POST",&format!("/v1/video/projects/{project}/edits"),Some(json!({"schema_version":1,"expected_project_revision":1,"commands":[{"operation":"set_creative","creative":{"references":[reference("keeper_id","keeper"),reference("visitor_id","visitor")],"sections":{}}}]}))).await;
    assert_eq!(code, 200, "{edited}");
    let mut start = request(&h, &project, "outline", "koboldcpp").await;
    start["cast"] = json!(["keeper_id", "visitor_id"]);
    start["images"] = json!([{"id":"source_setting","asset":asset,"label":"Own lighthouse","usage":"setting_reference"}]);
    start["add_dialogue"] = json!(true);
    state
        .fail_inventory_once
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let (code, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap();
    let failed = wait(&h, id, "failed").await;
    assert!(failed["outline"].is_null());
    assert_eq!(failed["inventory"], json!([]));
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/authoring/{id}/retry"),
            Some(json!({}))
        )
        .await
        .0,
        200
    );
    let outline = wait(&h, id, "outline").await;
    assert_eq!(outline["inventory"].as_array().unwrap().len(), 1);
    assert_eq!(
        outline["outline"]["scenes"][0]["image_ids"],
        json!(["source_setting"])
    );
    assert_eq!(outline["selected_cast"][0]["default_language"], "Swedish");
    assert!(!outline.to_string().contains("own-fixture-key"));
    {
        use base64::Engine;
        let requests = state.requests.lock().unwrap();
        let inventory = &requests[0]["messages"][1]["content"];
        let url = inventory[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"));
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(url.split_once(',').unwrap().1)
                .unwrap(),
            imagine_server::video::authoring::vision::normalize(&png())
                .unwrap()
                .jpeg
        );
    }
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/authoring/{id}/expand"),
            Some(json!({"schema_version":1,"start_index":0}))
        )
        .await
        .0,
        200
    );
    let ready = wait(&h, id, "review").await;
    let exact = ready["scenes"][0]["sections"][0]["prompt"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(exact.contains("@keeper says [Swedish] \"Hej!\""));
    assert_eq!(
        ready["scenes"][0]["sections"][0]["dialogue_intent"]["max_turns"],
        2
    );
    let (code, applied) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/apply"),
        Some(json!({"schema_version":1,"expected_project_revision":2})),
    )
    .await;
    assert_eq!(code, 200, "{applied}");
    assert_eq!(
        applied["project"]["scenes"][0]["sections"][0]["prompt"],
        exact
    );
    let section = applied["project"]["scenes"][0]["sections"][0]["id"]
        .as_str()
        .unwrap();
    assert_eq!(
        applied["project"]["creative"]["sections"][section]["images"][0]["asset"],
        asset
    );
    let before = state.requests.lock().unwrap().len();
    let mut enhancement = request(&h, &project, "enhance_section", "koboldcpp").await;
    enhancement["scene_id"] = applied["project"]["scenes"][0]["id"].clone();
    enhancement["section_id"] = json!(section);
    let (code, proposal) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(enhancement),
    )
    .await;
    assert_eq!(code, 202, "{proposal}");
    let enhancement = wait(&h, proposal["id"].as_str().unwrap(), "review").await;
    assert!(!enhancement["text"].as_str().unwrap().is_empty());
    {
        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), before + 1);
        assert!(
            requests.last().unwrap()["messages"][1]["content"].is_array(),
            "Section binding must reach the provider as an image, never silently fall back to text-only"
        );
    }
    let folder = root.path().join("project-owned");
    std::fs::create_dir(&folder).unwrap();
    let location = select(&h, &folder).await;
    let (code, saved) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/save"),
        Some(json!({"schema_version":1,"expected_project_revision":3,"location_id":location})),
    )
    .await;
    assert_eq!(code, 200, "{saved}");
    // Verify reopening after the owning service releases its autosave writer.
    stop(h).await;
    let reopened = imagine_server::video::persistence::ProjectSession::open(&folder).unwrap();
    assert_eq!(
        serde_json::to_value(&reopened.document().project).unwrap(),
        applied["project"]
    );
    assert_eq!(reopened.document().authoring[0].inventory.len(), 1);
    worker.abort();
    let _ = worker.await;
}
#[path = "../../examples/support/authoring_fixture.rs"]
pub(super) mod fixture;
async fn expanded(h: &Harness, project: &str, destination: &str) -> Value {
    let mut start = request(h, project, "outline", "koboldcpp").await;
    start["destination"] = json!(destination);
    let (code, d) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap();
    let mut d = wait(h, id, "outline").await;
    if destination == "continue" {
        d["outline"]["scenes"][0]["boundary_from_previous"] = json!("continuous_shot");
        assert_eq!(
            call(
                h,
                "POST",
                &format!("/v1/video/authoring/{id}/review"),
                Some(json!({"schema_version":1,"outline":d["outline"],"scenes":null,"text":null}))
            )
            .await
            .0,
            200
        );
    }
    assert_eq!(
        call(
            h,
            "POST",
            &format!("/v1/video/authoring/{id}/expand"),
            Some(json!({"schema_version":1,"start_index":0}))
        )
        .await
        .0,
        200
    );
    let ready = wait(h, id, "review").await;
    let(code,result)=call(h,"POST",&format!("/v1/video/authoring/{id}/apply"),Some(json!({"schema_version":1,"expected_project_revision":ready["request"]["expected_project_revision"]}))).await;
    assert_eq!(code, 200, "{result}");
    result
}
#[tokio::test]
async fn continuation_appends_after_occupied_end_and_preserves_reviewed_incoming_source() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "continue-story").await;
    let (url, worker, _) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let original = expanded(&h, &project, "append").await;
    let continued = expanded(&h, &project, "continue").await;
    assert_eq!(continued["project"]["scenes"].as_array().unwrap().len(), 4);
    assert_eq!(
        continued["project"]["scenes"][0],
        original["project"]["scenes"][0]
    );
    assert_eq!(
        continued["project"]["scenes"][1],
        original["project"]["scenes"][1]
    );
    assert_eq!(
        continued["project"]["scenes"][2]["start"],
        json!({"num":10,"den":1})
    );
    assert_eq!(
        continued["project"]["scenes"][2]["continuity"]["source_scene"],
        original["project"]["scenes"][1]["id"]
    );
    assert_eq!(
        continued["project"]["scenes"][2]["continuity"]["include_audio"],
        false
    );
    // A Story Continuous shot is the Timeline's: a continuous cut whose next
    // Scene carries the previous Scene's motion.
    assert_eq!(
        continued["project"]["scenes"][2]["continuity"]["method"],
        "motion_context_exp"
    );
    let cut = continued["project"]["assembly"]["boundaries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["right"] == continued["project"]["scenes"][2]["id"])
        .cloned()
        .expect("continuous cut");
    assert_eq!(cut["left"], original["project"]["scenes"][1]["id"]);
    assert_eq!(cut["mode"], "continuous_shot");
    let mut invalid = request(&h, &project, "outline", "koboldcpp").await;
    invalid["destination"] = json!("continue");
    let revision = continued["project"]["project_revision"].clone();
    let last = continued["project"]["scenes"][3]["id"].clone();
    assert_eq!(call(&h,"POST",&format!("/v1/video/projects/{project}/edits"),Some(json!({"schema_version":1,"expected_project_revision":revision,"commands":[{"operation":"set_scene_description","scene_id":last,"description":"x".repeat(16001)}]}))).await.0,200);
    invalid["expected_project_revision"] = json!(revision.as_u64().unwrap() + 1);
    let (code, error) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(invalid),
    )
    .await;
    assert_eq!(code, 422, "{error}");
    assert_eq!(error["error"]["code"], "story_continuation_context_limit");
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
async fn request(h: &Harness, project: &str, operation: &str, provider: &str) -> Value {
    let (_, d) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"provider_profile_id":"proxy","provider":provider,"prompt_profile_id":"normal","operation":operation,"brief":"Own synthetic lighthouse story","instruction":"","duration_seconds":10,"destination":"append"})
}
async fn configured(h: &Harness, url: &str) -> String {
    let (code, d) = call(h, "POST", "/v1/video/projects", Some(create())).await;
    assert_eq!(code, 201, "{d}");
    let profile = json!({"profile":{"id":"proxy","name":"Own fixture","base_url":url,"model":"own-story-fixture","system_prompt":"unused by video","max_tokens":1024,"temperature":null,"parameters":{},"has_api_key":false},"api_key":"own-fixture-key"});
    assert_eq!(
        call(h, "PUT", "/v1/enhancement/profiles/proxy", Some(profile))
            .await
            .0,
        200
    );
    d["project"]["id"].as_str().unwrap().to_owned()
}
async fn wait(h: &Harness, id: &str, phase: &str) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let (code, d) = call(h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
            assert_eq!(code, 200, "{d}");
            if d["phase"] == phase {
                return d;
            }
            assert_ne!(d["phase"], "failed", "{d}");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
#[tokio::test]
async fn story_snapshot_retry_review_atomic_apply_and_reload_preserve_provenance() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "story").await;
    let (url, worker, state) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let start = request(&h, &project, "outline", "koboldcpp").await;
    let (code, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    assert!(!d.to_string().contains("own-fixture-key"));
    let id = d["id"].as_str().unwrap();
    let outline = wait(&h, id, "outline").await;
    // Later provider edits must not affect the already admitted chain.
    let changed = json!({"profile":{"id":"proxy","name":"Changed later","base_url":"http://127.0.0.1:1/v1","model":"changed-model","system_prompt":"changed","max_tokens":1024,"temperature":null,"parameters":{}},"api_key":"changed-fixture-key"});
    assert_eq!(
        call(&h, "PUT", "/v1/enhancement/profiles/proxy", Some(changed))
            .await
            .0,
        200
    );
    state
        .fail_second_once
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let path = format!("/v1/video/authoring/{id}/expand");
    assert_eq!(
        call(
            &h,
            "POST",
            &path,
            Some(json!({"schema_version":1,"start_index":0}))
        )
        .await
        .0,
        200
    );
    let failed = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let (_, d) = call(&h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
            if d["phase"] == "failed" {
                break d;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(failed["failed_index"], 1);
    assert_eq!(failed["progress"]["completed"], 1);
    assert_eq!(failed["progress"]["total"], 2);
    assert!(!failed["scenes"][0].is_null());
    assert!(failed["scenes"][1].is_null());
    let first = failed["scenes"][0].clone();
    assert_eq!(
        call(
            &h,
            "POST",
            &path,
            Some(json!({"schema_version":1,"start_index":1}))
        )
        .await
        .0,
        200
    );
    let mut reviewed = wait(&h, id, "review").await;
    assert_eq!(reviewed["scenes"][0], first);
    reviewed["scenes"][1]["sections"][0]["prompt"] = json!("Own edited exact dialogue: Hej!");
    assert_eq!(call(&h,"POST",&format!("/v1/video/authoring/{id}/review"),Some(json!({"schema_version":1,"outline":reviewed["outline"],"scenes":reviewed["scenes"],"text":null}))).await.0,200);
    let (code,result)=call(&h,"POST",&format!("/v1/video/authoring/{id}/apply"),Some(json!({"schema_version":1,"expected_project_revision":outline["request"]["expected_project_revision"]}))).await;
    assert_eq!(code, 200, "{result}");
    assert_eq!(result["project"]["scenes"].as_array().unwrap().len(), 2);
    assert_eq!(result["authoring"][0]["phase"], "applied");
    assert_eq!(
        result["project"]["scenes"][1]["sections"][0]["prompt"],
        "Own edited exact dialogue: Hej!"
    );
    let second = &result["project"]["scenes"][1];
    assert_eq!(
        second["continuity"]["source_scene"],
        result["project"]["scenes"][0]["id"]
    );
    assert_eq!(second["continuity"]["include_audio"], false);
    assert_eq!(call(&h,"POST",&format!("/v1/video/authoring/{id}/apply"),Some(json!({"schema_version":1,"expected_project_revision":result["project"]["project_revision"]}))).await.0,409);
    let location = select(&h, &root.path().join("saved-story")).await;
    let (code,saved)=call(&h,"POST",&format!("/v1/video/projects/{project}/save-as"),Some(json!({"schema_version":1,"expected_project_revision":result["project"]["project_revision"],"location_id":location}))).await;
    assert_eq!(code, 201, "{saved}");
    assert_eq!(saved["authoring"], result["authoring"]);
    {
        let payloads = state.requests.lock().unwrap();
        assert_eq!(payloads.len(), 4);
        assert!(payloads.iter().all(|b| b["model"] == "own-story-fixture"));
        assert_eq!(payloads[0]["max_tokens"], 10000);
        assert_eq!(payloads[0]["temperature"], 0.05);
        assert_eq!(payloads[0]["chat_template_kwargs"]["enable_thinking"], true);
    }
    stop(h).await;
    let reopened = setup(root.path(), "story-reopened").await;
    let location = select(&reopened, &root.path().join("saved-story")).await;
    let (code, opened) = call(
        &reopened,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":location})),
    )
    .await;
    assert_eq!(code, 200, "{opened}");
    assert_eq!(opened["authoring"], saved["authoring"]);
    assert_eq!(opened["project"], saved["project"]);
    stop(reopened).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn stale_story_apply_is_blocked_and_cancellation_cannot_publish_late_result() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "stale-story").await;
    let (url, worker, _) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let start = request(&h, &project, "outline", "ollama").await;
    let (_, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    let id = d["id"].as_str().unwrap();
    let outline = wait(&h, id, "outline").await;
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/authoring/{id}/expand"),
            Some(json!({"schema_version":1,"start_index":0}))
        )
        .await
        .0,
        200
    );
    wait(&h, id, "review").await;
    let(code,d)=call(&h,"POST",&format!("/v1/video/projects/{project}/edits"),Some(json!({"schema_version":1,"expected_project_revision":outline["request"]["expected_project_revision"],"commands":[{"operation":"set_global_prompt","prompt":"Own concurrent edit"}]}))).await;
    assert_eq!(code, 200, "{d}");
    let(code,_) =call(&h,"POST",&format!("/v1/video/authoring/{id}/apply"),Some(json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"]}))).await;
    assert_eq!(code, 409);
    assert_eq!(
        wait(&h, id, "review").await["scenes"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let mut start = request(&h, &project, "outline", "koboldcpp").await;
    start["instruction"] = json!("fixture_wait");
    let (_, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    let id = d["id"].as_str().unwrap();
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/authoring/{id}/cancel"),
            Some(json!({}))
        )
        .await
        .1["phase"],
        "cancelled"
    );
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert_eq!(
        call(&h, "GET", &format!("/v1/video/authoring/{id}"), None)
            .await
            .1["phase"],
        "cancelled"
    );
    assert_eq!(
        call(&h, "GET", &format!("/v1/video/projects/{project}"), None)
            .await
            .1["project"]["scenes"],
        json!([])
    );
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn enhancements_review_section_and_scene_without_mutating_unselected_fields() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "enhance-story").await;
    let (url, worker, _) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let start = request(&h, &project, "outline", "koboldcpp").await;
    let (_, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    let id = d["id"].as_str().unwrap();
    let d = wait(&h, id, "outline").await;
    call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/expand"),
        Some(json!({"schema_version":1,"start_index":0})),
    )
    .await;
    wait(&h, id, "review").await;
    let(_,mut applied)=call(&h,"POST",&format!("/v1/video/authoring/{id}/apply"),Some(json!({"schema_version":1,"expected_project_revision":d["request"]["expected_project_revision"]}))).await;
    for operation in ["enhance_scene", "enhance_section"] {
        let before = applied.clone();
        let mut start = request(&h, &project, operation, "koboldcpp").await;
        start["scene_id"] = applied["project"]["scenes"][0]["id"].clone();
        start["section_id"] = applied["project"]["scenes"][0]["sections"][0]["id"].clone();
        let (code, d) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(start),
        )
        .await;
        assert_eq!(code, 202, "{d}");
        let id = d["id"].as_str().unwrap();
        let draft = wait(&h, id, "review").await;
        let text = "Own reviewed dialogue: Hej, världen!";
        assert_eq!(
            call(
                &h,
                "POST",
                &format!("/v1/video/authoring/{id}/review"),
                Some(json!({"schema_version":1,"outline":null,"scenes":null,"text":text}))
            )
            .await
            .0,
            200
        );
        let(code,result)=call(&h,"POST",&format!("/v1/video/authoring/{id}/apply"),Some(json!({"schema_version":1,"expected_project_revision":draft["request"]["expected_project_revision"]}))).await;
        assert_eq!(code, 200, "{result}");
        applied = result;
        assert_eq!(
            applied["project"]["scenes"][1],
            before["project"]["scenes"][1]
        );
        if operation == "enhance_scene" {
            assert_eq!(applied["project"]["scenes"][0]["description"], text);
            assert_eq!(
                applied["project"]["scenes"][0]["sections"],
                before["project"]["scenes"][0]["sections"]
            );
        } else {
            assert_eq!(
                applied["project"]["scenes"][0]["sections"][0]["prompt"],
                text
            );
            assert_eq!(
                applied["project"]["scenes"][0]["description"],
                before["project"]["scenes"][0]["description"]
            );
        }
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}

#[tokio::test]
async fn inline_enhancement_uses_and_applies_only_the_reviewed_scene_field() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "inline-fields").await;
    let (url, worker, state) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    expanded(&h, &project, "append").await;
    for field in [
        "summary",
        "ambient_sound",
        "non_diegetic_music",
        "description",
    ] {
        let (_, before) = call(&h, "GET", &format!("/v1/video/projects/{project}"), None).await;
        let scene = before["project"]["scenes"][0]["id"].as_str().unwrap();
        let source = if field == "description" {
            before["project"]["scenes"][0]["description"]
                .as_str()
                .unwrap_or("")
        } else {
            before["project"]["creative"]["scene_prompts"][scene][field]
                .as_str()
                .unwrap_or("")
        };
        let mut payload = request(&h, &project, "enhance_scene", "koboldcpp").await;
        payload["scene_id"] = json!(scene);
        payload["scene_field"] = json!(field);
        let (code, draft) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(payload),
        )
        .await;
        assert_eq!(code, 202, "{draft}");
        let id = draft["id"].as_str().unwrap();
        let draft = wait(&h, id, "review").await;
        assert_eq!(draft["request"]["scene_field"], field);
        let sent = state.requests.lock().unwrap().last().unwrap().clone();
        assert!(
            sent["messages"][1]["content"]
                .as_str()
                .unwrap()
                .contains(&format!("Source text:\n{source}"))
        );
        let text = format!("Own reviewed {field}");
        assert_eq!(
            call(
                &h,
                "POST",
                &format!("/v1/video/authoring/{id}/review"),
                Some(json!({"schema_version":1,"outline":null,"scenes":null,"text":text}))
            )
            .await
            .0,
            200
        );
        let(code,result)=call(&h,"POST",&format!("/v1/video/authoring/{id}/apply"),Some(json!({"schema_version":1,"expected_project_revision":draft["request"]["expected_project_revision"]}))).await;
        assert_eq!(code, 200, "{result}");
        let mut expected = before["project"].clone();
        if field == "description" {
            expected["scenes"][0]["description"] = json!(text);
        } else {
            expected["creative"]["scene_prompts"][scene][field] = json!(text);
        }
        expected["project_revision"] =
            json!(before["project"]["project_revision"].as_u64().unwrap() + 1);
        assert_eq!(result["project"], expected, "{field}");
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn regenerating_one_scene_keeps_the_others_and_a_failure_restores_it() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "regenerate").await;
    let (url, worker, state) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let start = request(&h, &project, "outline", "koboldcpp").await;
    let (code, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap();
    wait(&h, id, "outline").await;
    let expand = format!("/v1/video/authoring/{id}/expand");
    let body = |index: usize, only: bool| {
        Some(json!({"schema_version":1,"start_index":index,"only":only}))
    };
    assert_eq!(call(&h, "POST", &expand, body(0, false)).await.0, 200);
    let mut reviewed = wait(&h, id, "review").await;
    // Both Scenes are edited here; only the second is regenerated.
    reviewed["scenes"][0]["description"] = json!("Own kept first Scene");
    reviewed["scenes"][1]["description"] = json!("Own replaced second Scene");
    let review = format!("/v1/video/authoring/{id}/review");
    let edits = json!({"schema_version":1,"outline":reviewed["outline"],"scenes":reviewed["scenes"],"text":null});
    assert_eq!(call(&h, "POST", &review, Some(edits)).await.0, 200);
    let (code, running) = call(&h, "POST", &expand, body(1, true)).await;
    assert_eq!(code, 200, "{running}");
    assert!(running["scenes"][1].is_null() && !running["scenes"][0].is_null());
    let regenerated = wait(&h, id, "review").await;
    assert_eq!(
        regenerated["scenes"][0]["description"],
        "Own kept first Scene"
    );
    assert_ne!(
        regenerated["scenes"][1]["description"],
        "Own replaced second Scene"
    );
    assert!(
        regenerated["scenes"][1]["description"]
            .as_str()
            .unwrap()
            .starts_with("Expanded")
    );
    // A failed regeneration puts the Scene back and leaves the review open.
    state
        .fail_second_once
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(call(&h, "POST", &expand, body(1, true)).await.0, 200);
    let restored = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let (_, d) = call(&h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
            if d["failure"].is_string() && d["phase"] != "expanding" {
                break d;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(restored["phase"], "review");
    assert!(restored["failed_index"].is_null());
    assert_eq!(restored["scenes"][1], regenerated["scenes"][1]);
    assert_eq!(restored["scenes"][0], regenerated["scenes"][0]);
    worker.abort();
    let _ = worker.await;
    stop(h).await;
}
#[tokio::test]
async fn storyboard_review_accepts_cut_changes_but_not_structure_or_unknown_images() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "review-cuts").await;
    let (url, worker, _) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let start = request(&h, &project, "outline", "koboldcpp").await;
    let (_, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    let id = d["id"].as_str().unwrap();
    wait(&h, id, "outline").await;
    let expand = format!("/v1/video/authoring/{id}/expand");
    assert_eq!(
        call(
            &h,
            "POST",
            &expand,
            Some(json!({"schema_version":1,"start_index":0}))
        )
        .await
        .0,
        200
    );
    let ready = wait(&h, id, "review").await;
    let review = format!("/v1/video/authoring/{id}/review");
    let send = |scenes: Value| json!({"schema_version":1,"outline":ready["outline"],"scenes":scenes,"text":null});
    // The fixture's second Scene continues the first; review makes it a hard cut.
    let mut cut = ready["scenes"].clone();
    assert_eq!(cut[1]["boundary_from_previous"], "continuous_shot");
    cut[1]["boundary_from_previous"] = json!("hard_cut");
    let (code, saved) = call(&h, "POST", &review, Some(send(cut.clone()))).await;
    assert_eq!(code, 200, "{saved}");
    assert_eq!(saved["scenes"][1]["boundary_from_previous"], "hard_cut");
    let mut weight = cut.clone();
    weight[1]["weight"] = json!(5);
    let (code, error) = call(&h, "POST", &review, Some(send(weight))).await;
    assert_eq!(code, 422);
    assert_eq!(error["error"]["code"], "story_expansion_changed_structure");
    let mut image = cut.clone();
    image[0]["image_ids"] = json!(["own-unknown-image"]);
    let (code, error) = call(&h, "POST", &review, Some(send(image))).await;
    assert_eq!(code, 422);
    assert_eq!(error["error"]["code"], "story_image_missing");
    let (code, result) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/apply"),
        Some(json!({"schema_version":1,"expected_project_revision":ready["request"]["expected_project_revision"]})),
    )
    .await;
    assert_eq!(code, 200, "{result}");
    assert!(result["project"]["scenes"][1]["continuity"].is_null());
    assert!(
        result["project"]["assembly"]["boundaries"]
            .as_array()
            .is_none_or(|b| b.iter().all(|b| b["mode"] != "continuous_shot"))
    );
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn story_music_is_kept_only_when_asked_for() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "story-music").await;
    let (url, worker, _) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    for music in [false, true] {
        let mut start = request(&h, &project, "outline", "koboldcpp").await;
        start["add_music"] = json!(music);
        let (code, d) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(start),
        )
        .await;
        assert_eq!(code, 202, "{d}");
        let id = d["id"].as_str().unwrap();
        let outline = wait(&h, id, "outline").await;
        let expected = if music { "Own score" } else { "" };
        assert_eq!(
            outline["outline"]["scenes"][0]["non_diegetic_music"],
            expected
        );
        assert_eq!(
            call(
                &h,
                "POST",
                &format!("/v1/video/authoring/{id}/expand"),
                Some(json!({"schema_version":1,"start_index":0}))
            )
            .await
            .0,
            200
        );
        let ready = wait(&h, id, "review").await;
        assert_eq!(ready["scenes"][0]["non_diegetic_music"], expected);
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn whole_scene_enhancement_reviews_rows_side_by_side_and_applies_only_included_ones() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "enhance-all").await;
    let (url, worker, _) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let applied = expanded(&h, &project, "append").await;
    let before = applied["project"].clone();
    let scene = before["scenes"][0]["id"].as_str().unwrap().to_owned();
    let mut payload = request(&h, &project, "enhance_scene_all", "koboldcpp").await;
    payload["scene_id"] = json!(scene);
    let (code, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(payload),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap();
    let mut ready = wait(&h, id, "review").await;
    let rows = ready["fields"].as_array().unwrap().clone();
    let sections = before["scenes"][0]["sections"].as_array().unwrap().len();
    assert_eq!(rows.len(), 4 + sections);
    assert_eq!(
        rows[0]["target"],
        json!({"kind":"scene","field":"description"})
    );
    assert_eq!(rows[0]["original"], before["scenes"][0]["description"]);
    assert_eq!(rows[4]["target"]["kind"], "section");
    assert!(rows.iter().all(|r| {
        r["text"]
            .as_str()
            .is_some_and(|t| t.starts_with("Expanded"))
    }));
    assert_eq!(ready["progress"]["completed"], rows.len());
    // One row again, alone.
    let (code, again) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/expand"),
        Some(json!({"schema_version":1,"start_index":1,"only":true})),
    )
    .await;
    assert_eq!(code, 200, "{again}");
    assert!(again["fields"][1]["text"].is_null() && again["fields"][0]["text"].is_string());
    ready = wait(&h, id, "review").await;
    assert!(ready["fields"][1]["text"].is_string());
    // Edit the description, leave Summary out, keep Ambient's original.
    let mut fields = ready["fields"].clone();
    fields[0]["text"] = json!("Own reviewed description");
    fields[1]["include"] = json!(false);
    fields[2]["text"] = fields[2]["original"].clone();
    let review =
        json!({"schema_version":1,"outline":null,"scenes":null,"text":null,"fields":fields});
    let (code, saved) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/review"),
        Some(review),
    )
    .await;
    assert_eq!(code, 200, "{saved}");
    let mut changed = fields.clone();
    changed[0]["original"] = json!("Own changed original");
    let bad = json!({"schema_version":1,"outline":null,"scenes":null,"text":null,"fields":changed});
    let (code, error) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/review"),
        Some(bad),
    )
    .await;
    assert_eq!(code, 422);
    assert_eq!(error["error"]["code"], "invalid_authoring_fields");
    let (code, result) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/apply"),
        Some(json!({"schema_version":1,"expected_project_revision":ready["request"]["expected_project_revision"]})),
    )
    .await;
    assert_eq!(code, 200, "{result}");
    let after = &result["project"];
    assert_eq!(
        after["scenes"][0]["description"],
        "Own reviewed description"
    );
    let prompts = |p: &Value, field: &str| p["creative"]["scene_prompts"][&scene][field].clone();
    assert_eq!(prompts(after, "summary"), prompts(&before, "summary"));
    assert_eq!(
        prompts(after, "ambient_sound"),
        prompts(&before, "ambient_sound")
    );
    assert_eq!(
        after["scenes"][0]["sections"][0]["prompt"],
        ready["fields"][4]["text"]
    );
    // The other Scene is untouched.
    assert_eq!(after["scenes"][1], before["scenes"][1]);
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn enhancement_tells_the_model_the_first_frame_opens_the_scene() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "opening-frame").await;
    let (url, worker, state) = fixture::spawn().await;
    let project = configured(&h, &url).await;
    let (code, uploaded) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(png()),
        "image/png",
        true,
    )
    .await;
    assert_eq!(code, 201, "{uploaded}");
    let (code, attached) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":uploaded["id"]})),
    )
    .await;
    assert_eq!(code, 200, "{attached}");
    let asset = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    let section = |id: &str, start: u64| json!({"id":id,"start":{"num":start,"den":1},"duration":{"num":2,"den":1},"prompt":format!("Own {id} beat")});
    let scene = json!({"id":"own_scene","start":{"num":0,"den":1},"duration":{"num":4,"den":1},
        "description":"Own harbour at dusk","model":"project_default","settings":[],
        "sections":[section("own_first",0),section("own_second",2)],
        "bindings":[{"id":"own_binding","asset":asset,"role":"first_frame","enabled":true,"video":null}],
        "continuity":null,"accepted_take":null});
    let (code, added) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":1,"commands":[{"operation":"add_scene","scene":scene}]})),
    )
    .await;
    assert_eq!(code, 200, "{added}");
    let hint = "The first image is the Scene's opening frame";
    for (operation, field, section, opens) in [
        ("enhance_scene", Some("description"), None, true),
        ("enhance_section", None, Some("own_first"), true),
        ("enhance_section", None, Some("own_second"), false),
    ] {
        let mut payload = request(&h, &project, operation, "koboldcpp").await;
        payload["scene_id"] = json!("own_scene");
        payload["scene_field"] = json!(field);
        payload["section_id"] = json!(section);
        let (code, d) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(payload),
        )
        .await;
        assert_eq!(code, 202, "{d}");
        wait(&h, d["id"].as_str().unwrap(), "review").await;
        let sent = state.requests.lock().unwrap().last().unwrap().clone();
        let content = &sent["messages"][1]["content"];
        let text = content
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                content.as_array().map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<String>()
                })
            })
            .unwrap();
        assert_eq!(text.contains(hint), opens, "{operation} {section:?}");
        // The frame is still supplied as an image either way.
        assert!(
            text.contains("Images are supplied"),
            "{operation} {section:?}"
        );
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
