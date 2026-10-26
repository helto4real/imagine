use super::*;

async fn extra_scene(h: &Harness, project: &str, id: &str, source: Option<&str>) -> Value {
    let d = snapshot(h, project).await;
    let mut s = d["project"]["scenes"][0].clone();
    let index = d["project"]["scenes"].as_array().unwrap().len();
    s["id"] = json!(id);
    s["start"] = json!({"num":index,"den":1});
    s["accepted_take"] = Value::Null;
    s["description"] = json!(if source.is_some() {
        "own continuation target"
    } else {
        "own next scene"
    });
    if let Some(source) = source {
        let model = json!({"adapter":"h3","model":"minimax-h3","variant":"i2v","profile":imagine_server::video::job::I2V_PROFILE,"profile_revision":"2"});
        s["model"] = json!({"override":model});
        s["settings"][0]["model"] = model;
        s["continuity"] = json!({"source_scene":source,"method":"last_frame_first_frame"});
    }
    let (code,d)=call(h,"POST",&format!("/v1/video/projects/{project}/edits"),json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"commands":[{"operation":"add_scene","scene":s}]}),None).await;
    assert_eq!(code, 200, "{d}");
    d
}
async fn start(h: &Harness, project: &str, mode: &str, key: &str) -> Value {
    let d = snapshot(h, project).await;
    let request = json!({"schema_version":1,"project_id":project,"expected_project_revision":d["project"]["project_revision"],"mode":mode,"selected_scene":"s1"});
    let (code, run) = call(h, "POST", "/v1/video/runs", request.clone(), Some(key)).await;
    assert_eq!(code, 202, "{run}");
    let (code, replay) = call(h, "POST", "/v1/video/runs", request, Some(key)).await;
    assert_eq!(code, 202, "{replay}");
    assert_eq!(run["id"], replay["id"]);
    assert_eq!(replay["reused"], true);
    run
}
async fn terminal_run(h: &Harness, id: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let (code, run) =
                call(h, "GET", &format!("/v1/video/runs/{id}"), Value::Null, None).await;
            assert_eq!(code, 200, "{run}");
            if matches!(
                run["status"].as_str(),
                Some("completed" | "stopped" | "cancelled" | "failed" | "interrupted")
            ) {
                return run;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
async fn alter(h: &Harness, project: &str, scene: &str) {
    let d = snapshot(h, project).await;
    assert_eq!(call(h,"POST",&format!("/v1/video/projects/{project}/edits"),json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"commands":[{"operation":"set_scene_description","scene_id":scene,"description":"own changed during run"}]}),None).await.0,200);
}
#[tokio::test]
async fn run_is_sequential_uses_fifo_accepts_then_freezes_dependents_and_reuses_seeds() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "runs").await;
    let (p, _) = scene(&h).await;
    extra_scene(&h, &p, "s2", Some("s1")).await;
    extra_scene(&h, &p, "s3", Some("s2")).await;
    let run = start(&h, &p, "selected_with_dependents", "run-chain").await;
    assert_eq!(run["targets"].as_array().unwrap().len(), 3);
    let seeds = run["targets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["seed"].as_u64().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(seeds.len(), 3);
    wait_calls(&h, 1).await;
    let image = submit(&h, image("own image between run scenes"), "run-between").await;
    h.probe.release.notify_one();
    wait_calls(&h, 2).await;
    assert_eq!(
        h.probe.calls.lock().unwrap()[1],
        "own image between run scenes"
    );
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &image).await["status"], "succeeded");
    wait_calls(&h, 3).await;
    h.probe.release.notify_one();
    wait_calls(&h, 4).await;
    h.probe.release.notify_one();
    let done = terminal_run(&h, run["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "completed", "{done}");
    let d = snapshot(&h, &p).await;
    assert_eq!(d["takes"].as_array().unwrap().len(), 3);
    for (index, take) in d["takes"].as_array().unwrap().iter().enumerate() {
        assert_eq!(take["status"], "accepted");
        assert_eq!(take["provenance"]["seed"], run["targets"][index]["seed"]);
        assert_eq!(done["targets"][index]["completed_take_id"], take["id"]);
        if index > 0 {
            assert_eq!(
                take["provenance"]["source_take"],
                d["takes"][index - 1]["id"]
            );
        }
    }
    let (code,replayed)=call(&h,"POST","/v1/video/runs",json!({"schema_version":1,"project_id":p,"expected_project_revision":3,"mode":"selected_with_dependents","selected_scene":"s1"}),Some("run-chain")).await;
    assert_eq!(code, 202, "{replayed}");
    assert_eq!(replayed["id"], run["id"]);
    assert_eq!(replayed["status"], "completed");
    let same = start(&h, &p, "all_same_seeds", "run-same").await;
    for i in 0..3 {
        assert_eq!(same["targets"][i]["seed"], run["targets"][i]["seed"]);
        wait_calls(&h, 5 + i).await;
        h.probe.release.notify_one();
    }
    let done = terminal_run(&h, same["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "completed", "{done}");
    let skipped = start(&h, &p, "missing_only", "run-ready").await;
    assert_eq!(skipped["status"], "completed");
    assert!(skipped["targets"].as_array().unwrap().is_empty());
    stop(h).await;
}
#[tokio::test]
async fn run_errors_keep_completed_results_and_require_session_and_locked_seeds() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "run-errors").await;
    let (p, _) = scene(&h).await;
    extra_scene(&h, &p, "s2", None).await;
    let body = json!({"schema_version":1,"project_id":p,"expected_project_revision":2,"mode":"all_same_seeds","selected_scene":null});
    let (code, error) = call(&h, "POST", "/v1/video/runs", body, Some("no-seeds")).await;
    assert_eq!(code, 422);
    assert_eq!(error["error"]["code"], "all_scenes_require_locked_seeds");
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/video/runs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    h.probe.fail.store(true, Ordering::SeqCst);
    let r = start(&h, &p, "missing_only", "fail-run").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    let done = terminal_run(&h, r["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "failed");
    assert_eq!(done["failure"], "run_job_failed");
    assert_eq!(h.probe.calls.lock().unwrap().len(), 1);
    let d = snapshot(&h, &p).await;
    assert!(d["takes"].as_array().unwrap().is_empty());
    assert!(!d.to_string().contains(r["id"].as_str().unwrap()));
    stop(h).await;
}
#[tokio::test]
async fn stop_after_current_accepts_only_current_but_cancel_does_not_publish() {
    for cancel in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let h = setup(root.path(), "stop-run").await;
        let (p, _) = scene(&h).await;
        extra_scene(&h, &p, "s2", None).await;
        let run = start(&h, &p, "missing_only", "stop-run").await;
        wait_calls(&h, 1).await;
        let (code, _) = call(
            &h,
            "POST",
            &format!("/v1/video/runs/{}/stop", run["id"].as_str().unwrap()),
            json!({"schema_version":1,"cancel_current":cancel}),
            None,
        )
        .await;
        assert_eq!(code, 200);
        if !cancel {
            h.probe.release.notify_one();
        }
        let done = terminal_run(&h, run["id"].as_str().unwrap()).await;
        assert_eq!(
            done["status"],
            if cancel { "cancelled" } else { "stopped" },
            "{done}"
        );
        let d = snapshot(&h, &p).await;
        assert_eq!(d["takes"].as_array().unwrap().len(), usize::from(!cancel));
        assert_eq!(h.probe.calls.lock().unwrap().len(), 1);
        if !cancel {
            assert_eq!(d["takes"][0]["status"], "accepted");
        }
        stop(h).await;
    }
}
#[tokio::test]
async fn current_and_next_authoring_edits_cannot_be_overwritten_by_run() {
    for current in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let h = setup(root.path(), "run-edit").await;
        let (p, _) = scene(&h).await;
        extra_scene(&h, &p, "s2", None).await;
        let run = start(&h, &p, "missing_only", "run-edit").await;
        wait_calls(&h, 1).await;
        alter(&h, &p, if current { "s1" } else { "s2" }).await;
        h.probe.release.notify_one();
        let done = terminal_run(&h, run["id"].as_str().unwrap()).await;
        assert_eq!(done["status"], "failed");
        assert_eq!(done["failure"], "run_inputs_changed");
        let d = snapshot(&h, &p).await;
        assert_eq!(d["takes"].as_array().unwrap().len(), 1);
        assert_eq!(
            d["takes"][0]["status"],
            if current { "outdated" } else { "accepted" }
        );
        assert_eq!(h.probe.calls.lock().unwrap().len(), 1);
        stop(h).await;
    }
}
#[tokio::test]
async fn late_autoaccept_cannot_replace_a_manual_take_choice() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "run-choice").await;
    let (p, a) = scene(&h).await;
    let j = submit(&h, a, "before-run").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    terminal(&h, &j).await;
    let d = snapshot(&h, &p).await;
    let take = d["takes"][0]["id"].clone();
    assert_eq!(call(&h,"POST",&format!("/v1/video/projects/{p}/review"),json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"scene_id":"s1","take_id":take,"action":"accept"}),None).await.0,200);
    let run = start(&h, &p, "all_fresh_seeds", "choice").await;
    wait_calls(&h, 2).await;
    let d = snapshot(&h, &p).await;
    assert_eq!(call(&h,"POST",&format!("/v1/video/projects/{p}/review"),json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"scene_id":"s1","take_id":take,"action":"candidate"}),None).await.0,200);
    h.probe.release.notify_one();
    let done = terminal_run(&h, run["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "failed");
    assert_eq!(done["failure"], "run_take_selection_changed");
    let d = snapshot(&h, &p).await;
    assert!(d["project"]["scenes"][0]["accepted_take"].is_null());
    assert_eq!(d["takes"][1]["status"], "candidate");
    stop(h).await;
}
fn rerun(d: &Value, project: &str, rerun: Value) -> Value {
    json!({"schema_version":1,"project_id":project,"expected_project_revision":d["project"]["project_revision"],"mode":"rerun","rerun":rerun})
}
fn current(d: &Value, scene: &str) -> Value {
    d["project"]["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == scene)
        .unwrap()["accepted_take"]["id"]
        .clone()
}
fn take<'a>(d: &'a Value, id: &Value) -> &'a Value {
    d["takes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| &t["id"] == id)
        .unwrap()
}
async fn select(h: &Harness, project: &str, take: &Value, restore: bool) -> (u16, Value) {
    let d = snapshot(h, project).await;
    call(
        h,
        "POST",
        &format!(
            "/v1/video/projects/{project}/takes/{}/select",
            take.as_str().unwrap()
        ),
        json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"restore_prompt":restore}),
        None,
    )
    .await
}
#[tokio::test]
async fn rerun_renders_chosen_scenes_with_their_continuations_and_select_restores_the_chain() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "rerun").await;
    let (p, _) = scene(&h).await;
    extra_scene(&h, &p, "s2", Some("s1")).await;
    extra_scene(&h, &p, "s3", Some("s2")).await;
    extra_scene(&h, &p, "s4", None).await;
    let first = start(&h, &p, "missing_only", "rerun-first").await;
    for calls in 1..=4 {
        wait_calls(&h, calls).await;
        h.probe.release.notify_one();
    }
    assert_eq!(
        terminal_run(&h, first["id"].as_str().unwrap()).await["status"],
        "completed"
    );
    let d = snapshot(&h, &p).await;
    let old: Vec<_> = ["s1", "s2", "s3", "s4"]
        .iter()
        .map(|s| current(&d, s))
        .collect();
    let seed = |d: &Value, id: &Value| take(d, id)["provenance"]["seed"].clone();

    // The plan names the chosen scene and its continuation, and changes nothing.
    let (code, plan) = call(
        &h,
        "POST",
        "/v1/video/runs/plan",
        rerun(&d, &p, json!({"scenes":[{"scene_id":"s2"}]})),
        None,
    )
    .await;
    assert_eq!(code, 200, "{plan}");
    assert_eq!(plan["targets"].as_array().unwrap().len(), 2, "{plan}");
    assert_eq!(plan["targets"][0]["scene_id"], "s2");
    assert_eq!(plan["targets"][0]["reason"], "selected");
    assert_eq!(plan["targets"][0]["seed"], "fresh");
    assert_eq!(plan["targets"][0]["previous_take"], old[1]);
    assert_eq!(plan["targets"][1]["scene_id"], "s3");
    assert_eq!(plan["targets"][1]["reason"], "continues_from");
    assert_eq!(plan["targets"][1]["source"], "s2");
    assert_eq!(plan["targets"][1]["seed"], "locked_take");
    assert_eq!(plan["targets"][1]["seed_take_id"], old[2]);
    assert_eq!(plan["unchanged"], json!(["s1", "s4"]));
    assert_eq!(plan["total_seconds"], 2.0);
    assert_eq!(plan["warnings"], json!([]));

    // Requests that cannot run fail before any change.
    let s1_take = old[0].as_str().unwrap();
    for (body, code) in [
        (
            json!({"scenes":[{"scene_id":"s2","seed":"same"}]}),
            "rerun_would_repeat_take",
        ),
        (
            json!({"scenes":[{"scene_id":"s2"},{"scene_id":"s2"}]}),
            "rerun_scene_duplicate",
        ),
        (json!({"scenes":[]}), "rerun_scenes_required"),
        (
            json!({"scenes":[{"scene_id":"s2","seed":{"from_take":s1_take}}]}),
            "rerun_take_not_found",
        ),
        (
            json!({"scenes":[{"scene_id":"s2","prompt":{"sections":[{"section_id":"missing","prompt":"x"}]}}]}),
            "section_not_found",
        ),
        (
            json!({"scenes":[{"scene_id":"s2","prompt":{}}]}),
            "rerun_prompt_empty",
        ),
    ] {
        let (status, error) =
            call(&h, "POST", "/v1/video/runs/plan", rerun(&d, &p, body), None).await;
        assert_eq!((status, error["error"]["code"].as_str()), (422, Some(code)));
    }
    let mut missing = rerun(&d, &p, Value::Null);
    missing.as_object_mut().unwrap().remove("rerun");
    let (status, error) = call(&h, "POST", "/v1/video/runs/plan", missing, None).await;
    assert_eq!(
        (status, error["error"]["code"].as_str()),
        (422, Some("rerun_required"))
    );
    let mut wrong = rerun(&d, &p, json!({"scenes":[{"scene_id":"s2"}]}));
    wrong["mode"] = json!("missing_only");
    let (status, error) = call(&h, "POST", "/v1/video/runs/plan", wrong, None).await;
    assert_eq!(
        (status, error["error"]["code"].as_str()),
        (422, Some("rerun_requires_rerun_mode"))
    );

    // A changed prompt warns that the continuation's opening may no longer match.
    let changed = json!({"scenes":[{"scene_id":"s2","prompt":{"description":"own continuation target with a revised ending","summary":"own revised summary"}}]});
    let (_, plan) = call(
        &h,
        "POST",
        "/v1/video/runs/plan",
        rerun(&d, &p, changed.clone()),
        None,
    )
    .await;
    assert_eq!(plan["targets"][0]["prompt_changed"], true);
    assert_eq!(
        plan["warnings"],
        json!(["dependent_opening_may_mismatch:s3"])
    );
    assert_eq!(
        snapshot(&h, &p).await["project"]["project_revision"],
        d["project"]["project_revision"]
    );

    // Start commits the prompt with the run; the continuation keeps its seed.
    let (code, run) = call(
        &h,
        "POST",
        "/v1/video/runs",
        rerun(&d, &p, changed),
        Some("rerun-prompt"),
    )
    .await;
    assert_eq!(code, 202, "{run}");
    let edited = snapshot(&h, &p).await;
    assert_eq!(
        edited["project"]["project_revision"],
        d["project"]["project_revision"].as_u64().unwrap() + 1
    );
    assert_eq!(
        edited["project"]["scenes"][1]["description"],
        "own continuation target with a revised ending"
    );
    assert_eq!(
        edited["project"]["creative"]["scene_prompts"]["s2"]["summary"],
        "own revised summary"
    );
    assert_eq!(run["targets"][0]["reason"], "selected");
    assert_ne!(run["targets"][0]["seed"], seed(&d, &old[1]));
    assert_eq!(run["targets"][1]["seed"], seed(&d, &old[2]));
    for calls in 5..=6 {
        wait_calls(&h, calls).await;
        h.probe.release.notify_one();
    }
    assert_eq!(
        terminal_run(&h, run["id"].as_str().unwrap()).await["status"],
        "completed"
    );
    assert!(h.probe.calls.lock().unwrap()[4].contains("revised ending"));
    let d = snapshot(&h, &p).await;
    assert_eq!(d["takes"].as_array().unwrap().len(), 6);
    let new = [current(&d, "s2"), current(&d, "s3")];
    assert_eq!(current(&d, "s1"), old[0]);
    assert_eq!(current(&d, "s4"), old[3]);
    assert_ne!(new[0], old[1]);
    assert_eq!(take(&d, &old[1])["status"], "candidate");
    assert_eq!(take(&d, &old[2])["status"], "outdated");
    assert_eq!(take(&d, &new[1])["provenance"]["source_take"], new[0]);
    assert_eq!(
        d["take_states"][old[1].as_str().unwrap()]["inputs_current"],
        false
    );
    assert_eq!(
        d["take_states"][new[0].as_str().unwrap()]["inputs_current"],
        true
    );

    // Frames of a continuation start with its source's last frame.
    let (code, frames) = call(
        &h,
        "GET",
        &format!(
            "/v1/video/projects/{p}/takes/{}/frames",
            new[1].as_str().unwrap()
        ),
        Value::Null,
        None,
    )
    .await;
    assert_eq!(code, 200, "{frames}");
    let positions: Vec<_> = frames["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["position"].as_str().unwrap())
        .collect();
    assert_eq!(positions, ["source_last", "first", "middle", "last"]);
    assert_eq!(frames["frames"][0]["take_id"], new[0]);
    for frame in frames["frames"].as_array().unwrap() {
        use base64::Engine;
        let png = base64::engine::general_purpose::STANDARD
            .decode(frame["base64"].as_str().unwrap())
            .unwrap();
        let image = image::load_from_memory(&png).unwrap();
        assert_eq!((image.width(), image.height()), (32, 32));
    }

    // Going back to the first version needs its prompt and restores its continuation.
    let (code, error) = select(&h, &p, &old[1], false).await;
    assert_eq!(
        (code, error["error"]["code"].as_str()),
        (422, Some("take_prompt_changed"))
    );
    let (code, back) = select(&h, &p, &old[1], true).await;
    assert_eq!(code, 200, "{back}");
    assert_eq!(back["prompt_restored"], true);
    assert_eq!(
        back["restored"],
        json!([{"scene_id":"s3","take_id":old[2]}])
    );
    assert_eq!(back["needs_render"], json!([]));
    assert_eq!(back["final_video_outdated"], false);
    let d = snapshot(&h, &p).await;
    assert_eq!(
        [current(&d, "s2"), current(&d, "s3")],
        [old[1].clone(), old[2].clone()]
    );
    assert_eq!(
        d["project"]["scenes"][1]["description"],
        "own continuation target"
    );
    assert_eq!(
        d["project"]["creative"]["scene_prompts"]["s2"]["summary"],
        ""
    );
    let (code, error) = select(&h, &p, &old[1], false).await;
    assert_eq!(
        (code, error["error"]["code"].as_str()),
        (422, Some("take_already_current"))
    );

    // And forward again to the rerun's chain.
    let (code, forward) = select(&h, &p, &new[0], true).await;
    assert_eq!(code, 200, "{forward}");
    assert_eq!(
        forward["restored"],
        json!([{"scene_id":"s3","take_id":new[1]}])
    );
    assert_eq!(h.probe.calls.lock().unwrap().len(), 6);
    stop(h).await;
}
#[tokio::test]
async fn failed_rerun_keeps_finished_scenes_and_select_reports_continuations_to_render() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "rerun-fail").await;
    let (p, _) = scene(&h).await;
    extra_scene(&h, &p, "s2", Some("s1")).await;
    let first = start(&h, &p, "missing_only", "chain-first").await;
    for calls in 1..=2 {
        wait_calls(&h, calls).await;
        h.probe.release.notify_one();
    }
    terminal_run(&h, first["id"].as_str().unwrap()).await;
    let d = snapshot(&h, &p).await;
    let old = [current(&d, "s1"), current(&d, "s2")];

    // The continuation fails: the new first scene stays current, the continuation has none.
    let (code, run) = call(
        &h,
        "POST",
        "/v1/video/runs",
        rerun(&d, &p, json!({"scenes":[{"scene_id":"s1"}]})),
        Some("chain-fail"),
    )
    .await;
    assert_eq!(code, 202, "{run}");
    wait_calls(&h, 3).await;
    h.probe.release.notify_one();
    wait_calls(&h, 4).await;
    h.probe.fail.store(true, Ordering::SeqCst);
    h.probe.release.notify_one();
    let done = terminal_run(&h, run["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "failed", "{done}");
    let d = snapshot(&h, &p).await;
    let new = current(&d, "s1");
    assert_ne!(new, old[0]);
    assert!(current(&d, "s2").is_null());

    // Rerunning the continuation with its own seed completes the chain.
    let (code, plan) = call(
        &h,
        "POST",
        "/v1/video/runs/plan",
        rerun(&d, &p, json!({"scenes":[{"scene_id":"s2","seed":"same"}]})),
        None,
    )
    .await;
    assert_eq!(code, 200, "{plan}");
    assert_eq!(plan["targets"][0]["seed_take_id"], old[1]);

    // Without a kept continuation made from the old first take, select names it.
    let (code, _) = call(&h,"POST",&format!("/v1/video/projects/{p}/review"),json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"scene_id":"s2","take_id":old[1],"action":"reject"}),None).await;
    assert_eq!(code, 200);
    let (code, back) = select(&h, &p, &old[0], false).await;
    assert_eq!(code, 200, "{back}");
    assert_eq!(back["restored"], json!([]));
    assert_eq!(back["needs_render"], json!(["s2"]));
    assert_eq!(current(&snapshot(&h, &p).await, "s1"), old[0]);
    stop(h).await;
}
