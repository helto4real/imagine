//! Real resource freeze/decode/queue/publication with a synthetic sampler, no GPU.
use super::*;

#[tokio::test]
async fn transition_combo_influence_freeze_actual_take_tail_and_publish_applied_receipt() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "boundary").await;
    let (project, request) = scene(&h).await;
    let job = submit(&h, request, "boundary-source").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &job).await["status"], "succeeded");
    let mut document = snapshot(&h, &project).await;
    let take = document["takes"][0]["id"].clone();
    let (code,updated)=call(&h,"POST",&format!("/v1/video/projects/{project}/review"),
        json!({"schema_version":1,"expected_project_revision":document["project"]["project_revision"],
            "scene_id":"s1","take_id":take,"action":"accept"}),None).await;
    assert_eq!(code, 200, "{updated}");
    document = updated;
    let template = document["project"]["scenes"][0].clone();
    for (index, method) in ["combo", "influence"].into_iter().enumerate() {
        let id = format!("boundary_{method}");
        // The incoming seam is chronological even when the chosen media source
        // stays the earlier accepted s1 Take (the nonadjacent Influence case).
        let boundary_left = if index == 0 { "s1" } else { "boundary_combo" };
        let mut scene = template.clone();
        scene["id"] = json!(id);
        scene["start"] = json!({"num":index+1,"den":1});
        scene["accepted_take"] = Value::Null;
        scene["continuity"] = json!({"source_scene":"s1","method":method});
        let (code,updated)=call(&h,"POST",&format!("/v1/video/projects/{project}/edits"),
            json!({"schema_version":1,"expected_project_revision":document["project"]["project_revision"],
                "commands":[{"operation":"add_scene","scene":scene},
                    {"operation":"set_assembly_boundary","boundary":{"left":boundary_left,"right":id,
                        "mode":"transition","blend_frames":3,"transition_prompt":"The camera moves through the doorway.",
                        "audio_continuity":"reference_previous_audio"}}]}),None).await;
        assert_eq!(code, 200, "{updated}");
        document = updated;
        let job=submit(&h,json!({"kind":"video_scene","schema_version":1,"project_id":project,
            "expected_project_revision":document["project"]["project_revision"],"scene_id":id,"seed":42}),
            &format!("boundary-{method}")).await;
        wait_calls(&h, index + 2).await;
        {
            let captured = h.probe.references.lock().unwrap();
            let input = captured.last().unwrap();
            assert_eq!(input.videos.len(), 1);
            // H3's one-second selection resolves to its 39-frame shape.
            assert_eq!(input.videos[0].frames, 39);
            assert_eq!(input.videos[0].source_fps, 24.);
            assert!(input.videos[0].tail_frames.is_none());
            assert_eq!(input.videos[0].audio.as_ref().unwrap().samples, 71662);
            assert_eq!(
                input.manifest["boundary_selection"]["effective_tail_frames"],
                39
            );
            assert_eq!(input.manifest["videos"][0]["take_id"], take);
            assert_eq!(input.manifest["videos"][0]["prompt_label"], "<Video 1>");
            assert_eq!(
                input.manifest["audios"][0]["paired_video_key"],
                "ref_video_0"
            );
            if method == "combo" {
                assert!(input.images.is_empty());
                assert_eq!(input.manifest["frame0_guide"]["enabled"], true);
            } else {
                assert_eq!(input.images.len(), 1);
                assert_eq!(
                    input.images[0].rgb,
                    input.videos[0].rgb[input.videos[0].rgb.len() - 32 * 32 * 3..]
                );
            }
        }
        h.probe.release.notify_one();
        let completed = terminal(&h, &job).await;
        assert_eq!(completed["status"], "succeeded", "{completed}");
        document = snapshot(&h, &project).await;
        let receipt =
            &document["takes"].as_array().unwrap().last().unwrap()["provenance"]["boundary"];
        assert_eq!(receipt["applied"], true);
        assert_eq!(
            document["takes"].as_array().unwrap().last().unwrap()["provenance"]["execution"],
            json!({"attention":"auto","dynamic_vram":true})
        );
        assert_eq!(receipt["dependency"]["source_shot_id"], "s1");
        assert_eq!(receipt["dependency"]["boundary_id"]["left"], boundary_left);
        assert_eq!(receipt["dependency"]["target_shot_id"], id);
        assert_eq!(receipt["selection"]["effective_tail_frames"], 39);
        assert_eq!(document["takes"].as_array().unwrap().len(), index + 2);
    }
    stop(h).await;
}
