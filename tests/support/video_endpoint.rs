use super::*;

#[tokio::test]
async fn accepted_endpoint_drives_target_pixels_survives_reopen_and_source_replacement_invalidates()
{
    use imagine_server::video::{contract::ProjectId, persistence::ResourceKind};
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "endpoints1").await;
    let (project, source_request) = scene(&h).await;
    let source_job = submit(&h, source_request.clone(), "endpoint-source").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &source_job).await["status"], "succeeded");
    let document = snapshot(&h, &project).await;
    let source_take = document["takes"][0]["id"].clone();
    let endpoint = document["resources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|resource| resource["kind"]["kind"] == "h3_endpoint")
        .unwrap()
        .clone();
    assert_eq!(endpoint["kind"]["schema_version"], 1);
    assert_eq!(endpoint["kind"]["take_id"], source_take);
    let mut target_model = document["project"]["default_model"].clone();
    target_model["variant"] = json!("i2v");
    target_model["profile"] = json!(imagine_server::video::job::I2V_PROFILE);
    let target = json!({"id":"s2","start":{"num":1,"den":1},"duration":{"num":1,"den":1},"description":"own continuation target","model":{"override":target_model},"settings":[{"model":target_model,"schema_version":1,"values":{"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}}],"sections":[],"bindings":[],"continuity":{"source_scene":"s1","method":"last_frame_first_frame"},"accepted_take":null});
    let (code, added) = call(&h, "POST", &format!("/v1/video/projects/{project}/edits"), json!({"schema_version":1,"expected_project_revision":2,"commands":[{"operation":"add_scene","scene":target}]}), None).await;
    assert_eq!(code, 200, "{added}");
    let target_request = json!({"kind":"video_scene","schema_version":1,"project_id":project,"expected_project_revision":3,"scene_id":"s2","seed":42});
    let (code, missing) = call(
        &h,
        "POST",
        "/v1/jobs",
        target_request.clone(),
        Some("unaccepted-source"),
    )
    .await;
    assert_eq!(code, 422);
    assert_eq!(missing["error"]["code"], "continuity_source_not_accepted");
    let review = |revision, scene, take: &Value| json!({"schema_version":1,"expected_project_revision":revision,"scene_id":scene,"take_id":take,"action":"accept"});
    let route = format!("/v1/video/projects/{project}/review");
    assert_eq!(
        call(&h, "POST", &route, review(3, "s1", &source_take), None)
            .await
            .0,
        200
    );
    let mut target_request = target_request;
    target_request["expected_project_revision"] = json!(4);
    let target_job = submit(&h, target_request, "endpoint-target").await;
    wait_calls(&h, 2).await;
    h.probe.release.notify_one();
    let result = terminal(&h, &target_job).await;
    assert_eq!(result["status"], "succeeded", "{result}");
    let document = snapshot(&h, &project).await;
    let target_take = document["takes"][1]["id"].clone();
    let provenance = &document["takes"][1]["provenance"];
    assert_eq!(provenance["source_take"], source_take);
    assert_eq!(provenance["continuity"]["resource_id"], endpoint["id"]);
    assert_eq!(provenance["continuity"]["sha256"], endpoint["sha256"]);
    assert_eq!(provenance["continuity"]["method"], "last_frame_first_frame");
    assert_eq!(
        call(&h, "POST", &route, review(5, "s2", &target_take), None)
            .await
            .0,
        200
    );
    let (code, edited) = call(&h, "POST", &format!("/v1/video/projects/{project}/edits"), json!({"schema_version":1,"expected_project_revision":6,"commands":[{"operation":"set_scene_description","scene_id":"s1","description":"own changed source"}]}), None).await;
    assert_eq!(code, 200, "{edited}");
    assert_eq!(
        edited["project"]["scenes"][1]["accepted_take"]["id"],
        target_take
    );
    let mut again = source_request;
    again["expected_project_revision"] = json!(7);
    let job = submit(&h, again, "replacement-source").await;
    wait_calls(&h, 3).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &job).await["status"], "succeeded");
    let document = snapshot(&h, &project).await;
    let replacement = document["takes"][2]["id"].clone();
    let (code, replaced) = call(&h, "POST", &route, review(8, "s1", &replacement), None).await;
    assert_eq!(code, 200, "{replaced}");
    assert_eq!(replaced["takes"][1]["status"], "outdated");
    assert!(replaced["project"]["scenes"][1]["accepted_take"].is_null());
    assert_eq!(replaced["takes"][1]["provenance"], *provenance);
    let folder = root.path().join("endpoint-project");
    std::fs::create_dir(&folder).unwrap();
    let (_, location) = call(
        &h,
        "POST",
        "/v1/video/locations",
        json!({"schema_version":1,"path":folder}),
        None,
    )
    .await;
    let (code, saved) = call(&h, "POST", &format!("/v1/video/projects/{project}/save"), json!({"schema_version":1,"expected_project_revision":9,"location_id":location["location_id"]}), None).await;
    assert_eq!(code, 200, "{saved}");
    stop(h).await;
    let h = setup(root.path(), "endpoints2").await;
    let (_, location) = call(
        &h,
        "POST",
        "/v1/video/locations",
        json!({"schema_version":1,"path":folder}),
        None,
    )
    .await;
    let (code, reopened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        json!({"schema_version":1,"location_id":location["location_id"]}),
        None,
    )
    .await;
    assert_eq!(code, 200, "{reopened}");
    assert_eq!(reopened["takes"], saved["takes"]);
    let service = h.control.video.lock().await;
    let session = service
        .projects
        .session(&ProjectId(project.clone()))
        .unwrap();
    let endpoints = session
        .document()
        .resources
        .iter()
        .filter(|resource| matches!(resource.kind, ResourceKind::H3Endpoint { .. }))
        .count();
    assert_eq!(endpoints, 3);
    let mut png = vec![];
    session
        .copy_resource(endpoint["id"].as_str().unwrap(), &mut png)
        .unwrap();
    let rgb = image::load_from_memory(&png).unwrap().to_rgb8();
    assert_eq!(rgb.dimensions(), (32, 32));
    assert!(rgb.as_raw().iter().all(|pixel| *pixel == 70));
    drop(service);
    for mutation in ["version", "ambiguous", "missing", "unbound"] {
        let revision = {
            let mut service = h.control.video.lock().await;
            let session = service
                .projects
                .session_mut(&ProjectId(project.clone()))
                .unwrap();
            let mut document = session.document().clone();
            let selected = document.project.scenes[0]
                .accepted_take
                .as_ref()
                .unwrap()
                .id
                .clone();
            match mutation {
                "version" => {
                    for resource in &mut document.resources {
                        if let ResourceKind::H3Endpoint { take_id, schema_version, .. } = &mut resource.kind
                            && take_id == &selected { *schema_version = 2; }
                    }
                }
                "ambiguous" => {
                    let resource = document.resources.iter_mut().find(|resource| matches!(&resource.kind, ResourceKind::H3Endpoint { take_id, .. } if take_id == &selected)).unwrap();
                    if let ResourceKind::H3Endpoint { schema_version, .. } = &mut resource.kind { *schema_version = 1; }
                    let mut duplicate = resource.clone();
                    duplicate.id = "duplicate-owned-endpoint".into();
                    document.resources.push(duplicate);
                }
                "missing" => document.resources.retain(|resource| !matches!(&resource.kind, ResourceKind::H3Endpoint { take_id, .. } if take_id == &selected)),
                "unbound" => document.project.scenes[1].continuity.as_mut().unwrap().method = imagine_server::video::contract::ContinuityMethod::Combo,
                _ => unreachable!(),
            }
            let revision = document.project.project_revision;
            let staging =
                imagine_server::video::persistence::SessionStaging::select(root.path()).unwrap();
            session
                .update(revision, document, Default::default(), &staging)
                .unwrap();
            session.document().project.project_revision
        };
        let (code, rejected) = call(&h, "POST", "/v1/jobs", json!({"kind":"video_scene","schema_version":1,"project_id":project,"expected_project_revision":revision,"scene_id":"s2","seed":42}), Some(mutation)).await;
        assert_eq!(code, 422, "{mutation}: {rejected}");
        assert_eq!(
            rejected["error"]["code"],
            match mutation {
                "version" => "continuity_endpoint_incompatible",
                "unbound" => "continuity_method_unbound",
                _ => "continuity_endpoint_missing_or_ambiguous",
            }
        );
        assert!(h.probe.calls.lock().unwrap().is_empty());
    }
    stop(h).await;
}
