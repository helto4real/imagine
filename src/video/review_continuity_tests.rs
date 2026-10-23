//! Synthetic source/edit/freeze tests; no submitted job, decoder, model or worker.
use super::*;
use crate::video::{adapters::h3, execution, service::Service};
use imagine_video_contract::{
    assembly::{AudioContinuity, Boundary, BoundaryMode},
    boundary,
};
use std::{collections::BTreeMap, io::Write};

const ACTION: &str = "SYNTHETIC TARGET ACTION.";
const TRANSITION: &str = "The camera follows the synthetic subject through the doorway.";
const VISUAL: &str = "Visible subject identities, appearance, object state, and spatial relationships remain consistent at the seam until the authored action changes them.";
const STYLE: &str =
    "The opening preserves the established lighting, color, camera language, and visual style.";

fn tracked() -> ProjectDocument {
    let mut d = document();
    d.project.default_model.profile = crate::video::job::PROFILE.into();
    d.project.default_model.profile_revision = "2".into();
    d.project.delivery.width = 32;
    d.project.delivery.height = 32;
    for scene in &mut d.project.scenes {
        scene.continuity = None;
        scene.settings = vec![SettingsSlot {
            model: d.project.default_model.clone(),
            schema_version: 1,
            values: json!({"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}),
        }];
    }
    d.project.scenes[1].sections.push(Section {
        id: SectionId("own-continuity-section".into()),
        start: Time { num: 0, den: 1 },
        duration: Time { num: 2, den: 1 },
        prompt: ACTION.into(),
    });
    let mut seam = Boundary::new(SceneId("s0".into()), SceneId("s1".into()));
    seam.mode = BoundaryMode::Transition;
    seam.tail_frames = Some(23);
    seam.transition_prompt = TRANSITION.into();
    seam.audio_continuity = AudioContinuity::ReferencePreviousAudio;
    d.project.assembly.boundaries.push(seam);
    add_take(&mut d, 1, "own-unselected-candidate");
    let incoming = boundary::resolve(&d.project, &SceneId("s1".into()))
        .unwrap()
        .unwrap();
    d.takes[1].provenance.as_mut().unwrap().boundary =
        Some(json!({"dependency":boundary::dependency(&incoming,true),"applied":true}));
    d.validate().unwrap();
    d
}

fn off(mut seam: Boundary) -> Boundary {
    seam.reuse_visual_references = false;
    seam.reuse_style = false;
    seam.audio_continuity = AudioContinuity::NewSceneAudio;
    seam
}

#[test]
fn tracked_reuse_flags_reconcile_freeze_and_explicit_reuse_survives_reopen() {
    let original = tracked();
    // Each authored setting independently changes the tracked dependency. No
    // unrelated candidate, source selection or retained result is discarded.
    for field in 0..3 {
        let mut seam = original.project.assembly.boundaries[0].clone();
        match field {
            0 => seam.reuse_visual_references = false,
            1 => seam.reuse_style = false,
            _ => seam.audio_continuity = AudioContinuity::NewSceneAudio,
        }
        let changed = edit(
            &original,
            vec![Edit::SetAssemblyBoundary { boundary: seam }],
        );
        assert!(changed.project.scenes[1].accepted_take.is_none());
        assert_eq!(changed.takes[1].status, TakeStatus::Outdated);
        assert_eq!(changed.takes.last().unwrap().status, TakeStatus::Candidate);
        assert_eq!(changed.resources, original.resources);
        assert_eq!(
            changed.project.scenes[0].accepted_take,
            original.project.scenes[0].accepted_take
        );
    }

    let original_seam = original.project.assembly.boundaries[0].clone();
    let own_root = crate::task_fixtures::env_root("TMPDIR");
    let root = tempfile::Builder::new()
        .prefix("cp41continuity-")
        .tempdir_in(own_root)
        .unwrap();
    let staging = SessionStaging::select(root.path()).unwrap();
    let mut d = original;
    let mut files = BTreeMap::new();
    // These are opaque, explicitly constructed CPU transport bytes. They are
    // never decoded as video, nor produced by a model or submitted to a worker.
    let source_bytes = b"own synthetic source transport bytes: take t0";
    for (index, resource) in d.resources.iter_mut().enumerate() {
        let mut file = tempfile::tempfile_in(root.path()).unwrap();
        if index == 0 {
            file.write_all(source_bytes).unwrap();
        } else {
            file.write_all(format!("own synthetic retained transport bytes {index}").as_bytes())
                .unwrap();
        }
        *resource =
            Resource::inspect(resource.id.clone(), resource.kind.clone(), &mut file).unwrap();
        files.insert(resource.key(), file);
    }
    let id = d.project.id.clone();
    let revision = d.project.project_revision;
    let mut session = ProjectSession::new(d.project.clone()).unwrap();
    session.update(revision, d, files, &staging).unwrap();
    let saved = root.path().join("project");
    std::fs::create_dir(&saved).unwrap();
    session.save_to(&saved).unwrap();
    let mut service = Service::default();
    service.projects.insert(session).unwrap();
    let freeze = |service: &Service| {
        let d = service.projects.session(&id).unwrap().document();
        execution::freeze(
            service,
            &execution::Admission {
                kind: execution::Kind::VideoScene,
                schema_version: 1,
                project_id: id.clone(),
                expected_project_revision: d.project.project_revision,
                scene_id: SceneId("s1".into()),
                seed: 42,
            },
            &staging,
        )
        .unwrap()
    };
    let before = freeze(&service);
    assert_eq!(
        before.request.references.manifest["audios"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(
        before.request.prompt.contains(
            "The opening continues the ambience and physical sound character of <Audio 1>."
        )
    );
    let seam = off(service.projects.read(&id).unwrap().assembly.boundaries[0].clone());
    let revision = service.projects.read(&id).unwrap().project_revision;
    service
        .projects
        .edit(
            &id,
            EditRequest {
                schema_version: 1,
                expected_project_revision: revision,
                commands: vec![Edit::SetAssemblyBoundary { boundary: seam }],
            },
        )
        .unwrap();
    let frozen = freeze(&service);
    let manifest = &frozen.request.references.manifest;
    assert_eq!(manifest["videos"][0]["take_id"], "t0");
    assert_eq!(manifest["videos"][0]["source_scene_id"], "s0");
    assert_eq!(manifest["videos"][0]["requested_tail_frames"], 23);
    assert_eq!(manifest["videos"][0]["reuse_visual_references"], false);
    assert_eq!(manifest["videos"][0]["reuse_style"], false);
    assert_eq!(manifest["videos"][0]["include_paired_audio"], false);
    assert!(manifest["audios"].as_array().unwrap().is_empty());
    assert_eq!(manifest["frame0_guide"]["enabled"], true);
    assert_eq!(frozen.reference_inputs.len(), 1);
    assert_eq!(frozen.target.references.len(), 1);
    assert_eq!(frozen.target.references[0].0.id, "video_t0");
    let mut copied = vec![];
    crate::video::bundle::stream(&frozen.target.references[0].1, &mut copied).unwrap();
    assert_eq!(copied, source_bytes);
    let recipe = frozen.target.take.generated().unwrap();
    let receipt = recipe.boundary.as_ref().unwrap();
    assert_eq!(
        receipt["applied"], false,
        "freeze must not claim conditioning was executed"
    );
    assert_eq!(receipt["dependency"], manifest["boundary_dependency"]);
    assert_eq!(receipt["dependency"]["reuse_visual_references"], false);
    assert_eq!(receipt["dependency"]["reuse_style"], false);
    assert_eq!(receipt["dependency"]["audio_continuity"], "New Scene Audio");
    assert_eq!(recipe.source_take, Some(TakeId("t0".into())));
    assert!(frozen.request.prompt.contains(ACTION));
    assert!(frozen.request.prompt.contains(TRANSITION));
    assert!(
        frozen.request.prompt.contains(
            "The first frame of [Shot 1] is anchored to the previous Scene's final frame."
        )
    );
    assert!(
        !frozen.request.prompt.contains(
            "The opening continues the ambience and physical sound character of <Audio 1>."
        )
    );
    assert!(!frozen.request.prompt.contains(VISUAL));
    assert!(!frozen.request.prompt.contains(STYLE));
    assert_eq!(recipe.compiled_prompt, frozen.request.prompt);
    let reopened = ProjectSession::open(&saved).unwrap();
    assert_eq!(
        reopened.document(),
        service.projects.session(&id).unwrap().document()
    );
    assert!(
        reopened.document().project.scenes[1]
            .accepted_take
            .is_none()
    );
    // Director's Accept is followed by normalization/reconciliation: a stale
    // tracked receipt does not become current merely because it was selected.
    let old_receipt = reopened.document().takes[1]
        .generated()
        .unwrap()
        .boundary
        .clone();
    let mut reopened = reopened;
    reopened
        .review(request(reopened.document(), 1, "t1", ReviewAction::Accept))
        .unwrap();
    assert!(
        reopened.document().project.scenes[1]
            .accepted_take
            .is_none()
    );
    assert_eq!(reopened.document().takes[1].status, TakeStatus::Outdated);
    assert_eq!(
        reopened.document().takes[1].generated().unwrap().boundary,
        old_receipt
    );
    // Restoring the actual generation dependency permits explicit reuse again;
    // it never automatically accepts a retained result or rewrites its receipt.
    reopened
        .edit(EditRequest {
            schema_version: 1,
            expected_project_revision: reopened.document().project.project_revision,
            commands: vec![Edit::SetAssemblyBoundary {
                boundary: original_seam,
            }],
        })
        .unwrap();
    assert!(
        reopened.document().project.scenes[1]
            .accepted_take
            .is_none()
    );
    reopened
        .review(request(reopened.document(), 1, "t1", ReviewAction::Accept))
        .unwrap();
    assert_eq!(
        reopened.document().project.scenes[1]
            .accepted_take
            .as_ref()
            .unwrap()
            .id
            .0,
        "t1"
    );
    assert_eq!(
        reopened.document().takes[1].generated().unwrap().boundary,
        old_receipt
    );
    let reopened_again = ProjectSession::open(&saved).unwrap();
    assert_eq!(reopened_again.document(), reopened.document());
}

#[test]
fn reuse_prompt_effect_matches_seam_and_combo_influence_early_returns() {
    let input = |guide: bool, picture: bool, reuse: bool| h3::prompt::Input {
        sections: vec![
            json!({"item_id":"own_section","start_time":0.,"end_time":2.,"prompt":ACTION}),
        ],
        reference_manifest: json!({"subjects":[],"audios":[],
            "images":if picture {vec![json!({"role":"boundary_first_frame","prompt_label":"<Picture 1>"})]} else {vec![]},
            "videos":[{"role":"boundary_continuation","prompt_label":"<Video 1>","boundary_mode":"Transition",
                "transition_prompt":TRANSITION,"reuse_visual_references":reuse,"reuse_style":reuse}],
            "frame0_guide":{"enabled":guide}}),
        ..Default::default()
    };
    let with = h3::prompt::compile_reference(&input(false, false, true)).prompt;
    let without = h3::prompt::compile_reference(&input(false, false, false)).prompt;
    assert!(with.contains(VISUAL) && with.contains(STYLE));
    assert!(!without.contains(VISUAL) && !without.contains(STYLE));
    assert_eq!(with.replace(&format!(" {VISUAL} {STYLE}"), ""), without);
    // Director returns the anchor/picture opening before optional seam prose.
    for (guide, picture) in [(true, false), (false, true)] {
        assert_eq!(
            h3::prompt::compile_reference(&input(guide, picture, true)).prompt,
            h3::prompt::compile_reference(&input(guide, picture, false)).prompt
        );
    }
}
