use super::*;
#[tokio::test]
async fn exp_upscaled_source_tail_reaches_frozen_second_scene_with_same_handoff() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "exp-upscale").await;
    let (project, mut request) = scene(&h).await;
    let before = snapshot(&h, &project).await;
    let model = before["project"]["default_model"].clone();
    let settings = json!({"model":model,"schema_version":1,"values":{
        "stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},
        "stage_two":{"sampling":{"steps":8,"sampler":"euler","scheduler":"simple"},
            "model":model,"handoff":"re_noise_video_prediction","handoff_upscale_factor":2.,
            "trained_latent_upscale_model":"own-compatible-upscaler"}}});
    let route = format!("/v1/video/projects/{project}/edits");
    let (code, updated) = call(
        &h,
        "POST",
        &route,
        json!({"schema_version":1,
        "expected_project_revision":before["project"]["project_revision"],
        "commands":[{"operation":"set_model_settings","scene_id":"s1","settings":settings}]}),
        None,
    )
    .await;
    assert_eq!(code, 200, "{updated}");
    request["expected_project_revision"] = updated["project"]["project_revision"].clone();
    let source_job = submit(&h, request, "exp-upscale-source").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &source_job).await["status"], "succeeded");
    let document = snapshot(&h, &project).await;
    let source_take = document["takes"][0]["id"].clone();
    let resource = document["resources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["kind"]["kind"] == "h3_motion")
        .unwrap();
    assert_eq!(resource["kind"]["descriptor"]["shape"]["width"], 64);
    assert_eq!(
        resource["kind"]["descriptor"]["handoff_upscale"],
        json!({"base_width":32,"base_height":32,"factor":2.})
    );
    let (code,accepted)=call(&h,"POST",&format!("/v1/video/projects/{project}/review"),
        json!({"schema_version":1,"expected_project_revision":document["project"]["project_revision"],
        "scene_id":"s1","take_id":source_take,"action":"accept"}),None).await;
    assert_eq!(code, 200, "{accepted}");
    let mut target = accepted["project"]["scenes"][0].clone();
    target["id"] = json!("s2");
    target["start"] = json!({"num":1,"den":1});
    target["description"] = json!("own Exp upscale target");
    target["continuity"] =
        json!({"source_scene":"s1","method":"motion_context_exp","include_audio":true});
    target["accepted_take"] = Value::Null;
    let (code, updated) = call(
        &h,
        "POST",
        &route,
        json!({"schema_version":1,
        "expected_project_revision":accepted["project"]["project_revision"],
        "commands":[{"operation":"add_scene","scene":target}]}),
        None,
    )
    .await;
    assert_eq!(code, 200, "{updated}");
    let target_request = json!({"kind":"video_scene","schema_version":1,"project_id":project,
        "expected_project_revision":updated["project"]["project_revision"],"scene_id":"s2","seed":42});
    let target_job = submit(&h, target_request, "exp-upscale-target").await;
    wait_calls(&h, 2).await;
    h.probe.release.notify_one();
    let result = terminal(&h, &target_job).await;
    assert_eq!(result["status"], "succeeded", "{result}");
    let document = snapshot(&h, &project).await;
    let target = document["takes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|take| take["scene_id"] == "s2")
        .unwrap();
    assert_eq!(target["provenance"]["source_take"], source_take);
    assert_eq!(target["provenance"]["continuity"]["shape"]["width"], 64);
    assert_eq!(target["provenance"]["continuity"]["include_audio"], true);
    stop(h).await;
}
#[tokio::test]
async fn accepted_native_tail_is_frozen_for_both_motion_methods_with_explicit_audio_opt_in() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "motion").await;
    let (project, request) = scene(&h).await;
    let job = submit(&h, request, "motion-source").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &job).await["status"], "succeeded");
    let mut document = snapshot(&h, &project).await;
    let source_take = document["takes"][0]["id"].clone();
    let resource = document["resources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["kind"]["kind"] == "h3_motion")
        .unwrap()
        .clone();
    let template = document["project"]["scenes"][0].clone();
    let route = format!("/v1/video/projects/{project}/edits");
    for (index, (method, audio)) in [
        ("motion_context", false),
        ("motion_context", true),
        ("motion_context_exp", false),
        ("motion_context_exp", true),
    ]
    .into_iter()
    .enumerate()
    {
        let mut target = template.clone();
        let id = format!("motion_target_{index}");
        target["id"] = json!(id);
        target["start"] = json!({"num":index+1,"den":1});
        target["description"] = json!(format!("own {method} {audio}"));
        target["continuity"] = json!({"source_scene":"s1","method":method,"include_audio":audio});
        target["accepted_take"] = Value::Null;
        let (code,updated)=call(&h,"POST",&route,json!({"schema_version":1,"expected_project_revision":document["project"]["project_revision"],"commands":[{"operation":"add_scene","scene":target}]}),None).await;
        assert_eq!(code, 200, "{updated}");
        document = updated;
        let mut input = json!({"kind":"video_scene","schema_version":1,"project_id":project,"expected_project_revision":document["project"]["project_revision"],"scene_id":id,"seed":42});
        if index == 0 {
            let (code, missing) = call(
                &h,
                "POST",
                "/v1/jobs",
                input.clone(),
                Some("motion-unaccepted"),
            )
            .await;
            assert_eq!(code, 422);
            assert_eq!(missing["error"]["code"], "continuity_source_not_accepted");
            let (code,updated)=call(&h,"POST",&format!("/v1/video/projects/{project}/review"),json!({"schema_version":1,"expected_project_revision":document["project"]["project_revision"],"scene_id":"s1","take_id":source_take,"action":"accept"}),None).await;
            assert_eq!(code, 200, "{updated}");
            input["expected_project_revision"] = updated["project"]["project_revision"].clone();
        }
        let job = submit(&h, input, &format!("motion-{index}")).await;
        wait_calls(&h, index + 2).await;
        h.probe.release.notify_one();
        let result = terminal(&h, &job).await;
        assert_eq!(result["status"], "succeeded", "{result}");
        document = snapshot(&h, &project).await;
        let provenance = &document["takes"].as_array().unwrap().last().unwrap()["provenance"];
        assert_eq!(provenance["source_take"], source_take);
        assert_eq!(provenance["continuity"]["resource_id"], resource["id"]);
        assert_eq!(provenance["continuity"]["sha256"], resource["sha256"]);
        assert_eq!(provenance["continuity"]["method"], method);
        assert_eq!(provenance["continuity"]["include_audio"], audio);
    }
    let folder = root.path().join("motion-project");
    std::fs::create_dir(&folder).unwrap();
    let (code, location) = call(
        &h,
        "POST",
        "/v1/video/locations",
        json!({"schema_version":1,"path":folder}),
        None,
    )
    .await;
    assert_eq!(code, 201);
    let (code,saved)=call(&h,"POST",&format!("/v1/video/projects/{project}/save"),json!({"schema_version":1,"expected_project_revision":document["project"]["project_revision"],"location_id":location["location_id"]}),None).await;
    assert_eq!(code, 200, "{saved}");
    stop(h).await;
    let reopened = setup(root.path(), "motion-reopen").await;
    let (_, location) = call(
        &reopened,
        "POST",
        "/v1/video/locations",
        json!({"schema_version":1,"path":folder}),
        None,
    )
    .await;
    let (code, opened) = call(
        &reopened,
        "POST",
        "/v1/video/projects/open",
        json!({"schema_version":1,"location_id":location["location_id"]}),
        None,
    )
    .await;
    assert_eq!(code, 200, "{opened}");
    assert_eq!(opened["takes"], saved["takes"]);
    assert_eq!(opened["resources"], saved["resources"]);
    stop(reopened).await;
}
