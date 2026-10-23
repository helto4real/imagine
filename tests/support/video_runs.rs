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
