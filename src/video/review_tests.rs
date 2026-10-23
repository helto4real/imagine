use super::super::editing::{self, Edit, EditRequest};
use super::super::planning;
use super::*;
use serde_json::json;

#[test]
fn locked_take_index_survives_invalidation_and_legacy_documents_remain_readable() {
    let mut d = document();
    assert_eq!(
        d.locked_takes.get(&SceneId("s1".into())),
        Some(&TakeId("t1".into()))
    );
    let mut legacy = serde_json::to_value(&d).unwrap();
    legacy.as_object_mut().unwrap().remove("locked_takes");
    let legacy: ProjectDocument = serde_json::from_value(legacy).unwrap();
    legacy.validate().unwrap();
    assert!(legacy.locked_takes.is_empty());
    add_take(&mut d, 0, "replacement");
    let mut changed = review(&d, 0, "replacement", ReviewAction::Accept).unwrap();
    assert!(changed.project.scenes[1].accepted_take.is_none());
    assert_eq!(
        changed.locked_takes.get(&SceneId("s1".into())),
        Some(&TakeId("t1".into()))
    );
    let saved = serde_json::to_vec(&changed).unwrap();
    let loaded: ProjectDocument = serde_json::from_slice(&saved).unwrap();
    loaded.validate().unwrap();
    assert_eq!(loaded.locked_takes, changed.locked_takes);
    changed
        .locked_takes
        .insert(SceneId("s1".into()), TakeId("t0".into()));
    assert_eq!(changed.validate().unwrap_err().code, "locked_take_mismatch");
}

fn document() -> ProjectDocument {
    let mut project = VideoProject::create(
        serde_json::from_str(include_str!(
            "../../design/video-contract/create-project.json"
        ))
        .unwrap(),
    )
    .unwrap();
    for index in 0..3 {
        project.scenes.push(Scene {
            id: SceneId(format!("s{index}")),
            start: Time {
                num: index * 2,
                den: 1,
            },
            duration: Time { num: 2, den: 1 },
            description: format!("Synthetic scene {index}"),
            model: ModelChoice::ProjectDefault,
            settings: vec![SettingsSlot {
                model: project.default_model.clone(),
                schema_version: 1,
                values: json!({}),
            }],
            sections: vec![],
            bindings: vec![],
            accepted_take: None,
            continuity: (index > 0).then(|| Continuity {
                include_audio: false,
                source_scene: SceneId(format!("s{}", index - 1)),
                method: ContinuityMethod::LastFrameFirstFrame,
            }),
        });
    }
    let mut document = ProjectDocument::new(project);
    for index in 0..3 {
        add_take(&mut document, index, &format!("t{index}"));
        document = review(&document, index, &format!("t{index}"), ReviewAction::Accept).unwrap();
    }
    document
}
fn add_take(document: &mut ProjectDocument, scene: u64, id: &str) {
    let scene_id = SceneId(format!("s{scene}"));
    let model = document.project.default_model.clone();
    let source_take = document
        .project
        .scene(&scene_id)
        .unwrap()
        .continuity
        .as_ref()
        .and_then(|continuity| {
            document
                .project
                .scene(&continuity.source_scene)
                .unwrap()
                .accepted_take
                .as_ref()
                .map(|take| take.id.clone())
        });
    document.takes.push(StoredTake {
        id: TakeId(id.into()),
        scene_id: scene_id.clone(),
        status: TakeStatus::Candidate,
        provenance: TakeProvenance {
            model: model.clone(),
            seed: 42,
            input_revision: planning::input_revision(&document.project, &scene_id).unwrap(),
            compiled_prompt: "Synthetic fixture".into(),
            settings: SettingsSlot {
                model,
                schema_version: 1,
                values: json!({}),
            },
            references: vec![],
            source_take,
            continuity: None,
        },
    });
    document.resources.push(Resource {
        id: format!("video_{id}"),
        kind: ResourceKind::TakeVideo {
            take_id: TakeId(id.into()),
        },
        sha256: "a".repeat(64),
        bytes: 1,
    });
}
fn request(
    document: &ProjectDocument,
    scene: u64,
    take: &str,
    action: ReviewAction,
) -> ReviewRequest {
    ReviewRequest {
        schema_version: 1,
        expected_project_revision: document.project.project_revision,
        scene_id: SceneId(format!("s{scene}")),
        take_id: TakeId(take.into()),
        action,
    }
}
fn review(
    document: &ProjectDocument,
    scene: u64,
    take: &str,
    action: ReviewAction,
) -> Result<ProjectDocument, StorageError> {
    apply(document, request(document, scene, take, action))
}
fn edit(document: &ProjectDocument, commands: Vec<Edit>) -> ProjectDocument {
    let mut candidate = document.clone();
    editing::apply(
        &mut candidate.project,
        EditRequest {
            schema_version: 1,
            expected_project_revision: document.project.project_revision,
            commands,
        },
    )
    .unwrap();
    invalidate_changed(&document.project, &mut candidate).unwrap();
    candidate.validate().unwrap();
    candidate
}

#[test]
fn changed_source_invalidates_every_descendant_but_keeps_results_and_rejections() {
    let mut original = document();
    add_take(&mut original, 1, "rejected");
    original = review(&original, 1, "rejected", ReviewAction::Reject).unwrap();
    let changed = edit(
        &original,
        vec![Edit::SetContinuity {
            scene_id: SceneId("s1".into()),
            continuity: Some(Continuity {
                include_audio: false,
                source_scene: SceneId("s0".into()),
                method: ContinuityMethod::Combo,
            }),
        }],
    );
    assert!(
        changed
            .project
            .scenes
            .iter()
            .skip(1)
            .all(|scene| scene.accepted_take.is_none())
    );
    assert_eq!(changed.takes.len(), 4);
    assert!(
        changed.takes[1..3]
            .iter()
            .all(|take| take.status == TakeStatus::Outdated)
    );
    assert_eq!(changed.takes[3].status, TakeStatus::Rejected);
    assert_eq!(changed.resources, original.resources);
    assert!(
        original
            .project
            .scenes
            .iter()
            .all(|scene| scene.accepted_take.is_some())
    );
}

#[test]
fn choosing_another_source_invalidates_descendants_and_old_take_is_an_explicit_choice() {
    let mut original = document();
    add_take(&mut original, 0, "alternate");
    let changed = review(&original, 0, "alternate", ReviewAction::Accept).unwrap();
    assert_eq!(changed.takes[0].status, TakeStatus::Candidate);
    assert_eq!(changed.takes[3].status, TakeStatus::Accepted);
    assert!(
        changed.project.scenes[1..]
            .iter()
            .all(|scene| scene.accepted_take.is_none())
    );
    let old = review(&changed, 1, "t1", ReviewAction::Accept).unwrap();
    assert!(states(&old)[&TakeId("t1".into())].accepted);
    assert!(!states(&old)[&TakeId("t1".into())].outdated);
    let same = edit(
        &old,
        vec![Edit::SetGlobalPrompt {
            prompt: old.project.global_prompt.clone(),
        }],
    );
    assert_eq!(
        same.project.scenes[1].accepted_take,
        old.project.scenes[1].accepted_take
    );
    let changed_again = edit(
        &old,
        vec![Edit::SetSceneDescription {
            scene_id: SceneId("s1".into()),
            description: "Changed again".into(),
        }],
    );
    assert_eq!(
        changed_again.project.scenes[1].accepted_take,
        old.project.scenes[1].accepted_take
    );
}

#[test]
fn reviewing_nonselected_candidate_preserves_selection_and_reverting_never_autoaccepts() {
    let mut original = document();
    add_take(&mut original, 0, "other");
    let rejected = review(&original, 0, "other", ReviewAction::Reject).unwrap();
    assert_eq!(
        rejected.project.scenes[0].accepted_take,
        original.project.scenes[0].accepted_take
    );
    assert_eq!(rejected.takes[0].status, TakeStatus::Accepted);
    let removed = review(&rejected, 0, "t0", ReviewAction::Candidate).unwrap();
    assert!(
        removed
            .project
            .scenes
            .iter()
            .all(|scene| scene.accepted_take.is_none())
    );
    let reverted = review(&removed, 0, "t0", ReviewAction::Accept).unwrap();
    assert_eq!(reverted.takes[0].status, TakeStatus::Accepted);
    assert!(
        reverted.project.scenes[1..]
            .iter()
            .all(|scene| scene.accepted_take.is_none())
    );
    assert!(
        reverted.takes[1..3]
            .iter()
            .all(|take| take.status == TakeStatus::Outdated)
    );
}

#[test]
fn final_mix_and_inactive_model_settings_do_not_invalidate_generation() {
    let original = document();
    let mut settings = original.project.audio.clone();
    settings.use_native_audio = !settings.use_native_audio;
    let mut model = original.project.default_model.clone();
    model.profile = "inactive-profile".into();
    let changed = edit(
        &original,
        vec![
            Edit::SetSceneDescription {
                scene_id: SceneId("s0".into()),
                description: "Changed prompt".into(),
            },
            Edit::SetDefaultModel {
                model: model.clone(),
            },
            Edit::SetAudioSettings { settings },
            Edit::SetModelSettings {
                scene_id: SceneId("s0".into()),
                settings: SettingsSlot {
                    model,
                    schema_version: 1,
                    values: json!({"inactive":true}),
                },
            },
        ],
    );
    assert_eq!(changed.takes, original.takes);
    assert_eq!(
        changed.project.scenes[0].accepted_take,
        original.project.scenes[0].accepted_take
    );
}

#[test]
fn review_rejects_conflicts_cross_scene_missing_resource_and_revision_overflow_atomically() {
    let original = document();
    let mut stale = request(&original, 0, "t0", ReviewAction::Reject);
    stale.expected_project_revision -= 1;
    assert_eq!(
        apply(&original, stale).unwrap_err().code,
        "revision_conflict"
    );
    assert_eq!(
        review(&original, 1, "t0", ReviewAction::Accept)
            .unwrap_err()
            .code,
        "take_not_found"
    );
    let mut missing = original.clone();
    missing.resources.clear();
    assert_eq!(
        review(&missing, 0, "t0", ReviewAction::Accept)
            .unwrap_err()
            .code,
        "take_resource_missing"
    );
    let mut overflow = original.clone();
    overflow.project.project_revision = MAX_INTEGER;
    assert_eq!(
        review(&overflow, 0, "t0", ReviewAction::Reject)
            .unwrap_err()
            .code,
        "invalid_revision"
    );
    assert!(
        original
            .project
            .scenes
            .iter()
            .all(|scene| scene.accepted_take.is_some())
    );
}
