use super::super::editing::{self, Edit, EditRequest};
use super::super::planning;
use super::*;
use serde_json::json;

#[path = "review_continuity_tests.rs"]
mod continuity;

#[test]
fn boundary_reconciliation_only_invalidates_tracked_selected_take_on_generation_changes() {
    use imagine_video_contract::{
        assembly::{Boundary, BoundaryMode},
        boundary,
    };
    let mut d = document();
    d.project.scenes[1].continuity = None;
    let mut seam = Boundary::new(SceneId("s0".into()), SceneId("s1".into()));
    seam.mode = BoundaryMode::Transition;
    d.project.assembly.boundaries.push(seam);
    add_take(&mut d, 1, "unrelated-candidate");
    let incoming = boundary::resolve(&d.project, &SceneId("s1".into()))
        .unwrap()
        .unwrap();
    d.takes
        .iter_mut()
        .find(|t| t.id.0 == "t1")
        .unwrap()
        .provenance
        .as_mut()
        .unwrap()
        .boundary = Some(json!({"dependency":boundary::dependency(&incoming,true),"applied":true}));
    let legacy = d.clone();
    let before = d.project.clone();
    d.project.assembly.boundaries[0].blend_frames = 12;
    invalidate_changed(&before, &mut d).unwrap();
    assert_eq!(
        d.project.scenes[1].accepted_take.as_ref().unwrap().id.0,
        "t1"
    );
    let before = d.project.clone();
    d.project.assembly.boundaries[0].transition_prompt = "The camera turns.".into();
    invalidate_changed(&before, &mut d).unwrap();
    assert!(d.project.scenes[1].accepted_take.is_none());
    assert_eq!(
        d.takes.iter().find(|t| t.id.0 == "t1").unwrap().status,
        TakeStatus::Outdated
    );
    assert_eq!(
        d.takes
            .iter()
            .find(|t| t.id.0 == "unrelated-candidate")
            .unwrap()
            .status,
        TakeStatus::Candidate
    );
    let mut untracked = legacy;
    untracked
        .takes
        .iter_mut()
        .find(|t| t.id.0 == "t1")
        .unwrap()
        .provenance
        .as_mut()
        .unwrap()
        .boundary = None;
    let before = untracked.project.clone();
    untracked.project.assembly.boundaries[0].transition_prompt = "New request.".into();
    invalidate_changed(&before, &mut untracked).unwrap();
    assert_eq!(
        untracked.project.scenes[1]
            .accepted_take
            .as_ref()
            .unwrap()
            .id
            .0,
        "t1"
    );
}

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
        provenance: Some(TakeProvenance {
            boundary: None,
            resolved_loras: None,
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
        }),
    });
    document.resources.push(Resource {
        id: format!("video_{id}"),
        source: None,
        kind: ResourceKind::TakeVideo {
            take_id: TakeId(id.into()),
        },
        sha256: "a".repeat(64),
        bytes: 1,
    });
}

#[test]
fn external_video_take_has_content_identity_without_a_fabricated_recipe() {
    use crate::video::captures::{self, Action, Import};
    let mut original = document();
    let asset_id = AssetId("own-external-video".into());
    original.project.assets.insert(
        asset_id.clone(),
        Asset {
            id: asset_id.clone(),
            name: "Own video".into(),
            revision: 1,
            kind: MediaKind::Video,
            library_source: None,
        },
    );
    original.resources.push(Resource {
        id: "own-external-resource".into(),
        source: None,
        kind: ResourceKind::Reference {
            asset_id: asset_id.clone(),
            revision: 1,
        },
        sha256: "b".repeat(64),
        bytes: 123,
    });
    let make = |d: &ProjectDocument| Import {
        schema_version: 1,
        expected_project_revision: d.project.project_revision,
        scene_id: SceneId("s0".into()),
        asset_id: asset_id.clone(),
        accept: true,
    };
    let imported = captures::import(&original, make(&original)).unwrap();
    let id = imported.attached_id.clone().unwrap();
    let take = imported.document.takes.iter().find(|t| t.id == id).unwrap();
    assert!(take.provenance.is_none());
    let accepted = imported.document.project.scenes[0]
        .accepted_take
        .as_ref()
        .unwrap();
    assert_eq!(accepted.id, id);
    assert!(accepted.model.is_none());
    assert_eq!(accepted.source_fingerprint, "b".repeat(64));
    assert_eq!(
        take.generated().unwrap_err().code,
        "take_generation_recipe_unknown"
    );
    let encoded = serde_json::to_value(&imported.document).unwrap();
    assert!(
        encoded["takes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["id"] == id.0)
            .unwrap()["provenance"]
            .is_null()
    );
    // Existing generated recipes retain their object-shaped wire representation.
    assert!(encoded["takes"][0]["provenance"].is_object());
    let loaded: ProjectDocument = serde_json::from_value(encoded).unwrap();
    loaded.validate().unwrap();
    let again = captures::import(&loaded, make(&loaded)).unwrap();
    assert_eq!(again.attached_id, Some(id.clone()));
    assert_eq!(again.document.takes.len(), loaded.takes.len());
    let copied = captures::apply(
        &again.document,
        capture_request(&again.document, 2, &id.0, Action::Attach),
    )
    .unwrap();
    let mut in_second_scene = make(&copied.document);
    in_second_scene.scene_id = SceneId("s2".into());
    let selected_again = captures::import(&copied.document, in_second_scene).unwrap();
    assert_eq!(selected_again.attached_id, copied.attached_id);
    assert_eq!(
        selected_again.document.takes.len(),
        copied.document.takes.len()
    );
    let deleted = captures::apply(
        &again.document,
        capture_request(&again.document, 0, &id.0, Action::Delete),
    )
    .unwrap();
    assert!(
        deleted
            .document
            .resources
            .iter()
            .any(|r| r.id == "own-external-resource")
    );
    assert!(deleted.document.project.assets.contains_key(&asset_id));
    let mut wrong_kind = original.clone();
    wrong_kind.project.assets.get_mut(&asset_id).unwrap().kind = MediaKind::Audio;
    assert_eq!(
        captures::import(&wrong_kind, make(&wrong_kind))
            .err()
            .unwrap()
            .code,
        "take_requires_video"
    );
    assert_eq!(original.takes.len(), 3);
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

fn capture_request(
    d: &ProjectDocument,
    scene: u64,
    take: &str,
    action: super::super::captures::Action,
) -> super::super::captures::Request {
    super::super::captures::Request {
        schema_version: 1,
        expected_project_revision: d.project.project_revision,
        scene_id: SceneId(format!("s{scene}")),
        take_id: TakeId(take.into()),
        action,
    }
}

#[test]
fn capture_detach_retains_receipt_and_media_reattach_accept_restores_selection() {
    use super::super::captures::{self, Action};
    let original = document();
    let detached = captures::apply(
        &original,
        capture_request(&original, 0, "t0", Action::Detach),
    )
    .unwrap()
    .document;
    assert!(!captures::attached(&detached, &TakeId("t0".into())));
    assert!(detached.project.scenes[0].accepted_take.is_none());
    assert_eq!(detached.takes[0].provenance, original.takes[0].provenance);
    assert_eq!(detached.resources, original.resources);
    assert!(detached.project.scenes[1].accepted_take.is_none());
    assert_eq!(
        review(&detached, 0, "t0", ReviewAction::Accept)
            .unwrap_err()
            .code,
        "take_not_attached"
    );
    let restored = captures::apply(
        &detached,
        capture_request(&detached, 0, "t0", Action::AttachAndAccept),
    )
    .unwrap()
    .document;
    assert!(captures::attached(&restored, &TakeId("t0".into())));
    assert_eq!(
        restored.project.scenes[0]
            .accepted_take
            .as_ref()
            .unwrap()
            .id
            .0,
        "t0"
    );
    let loaded: ProjectDocument =
        serde_json::from_slice(&serde_json::to_vec(&restored).unwrap()).unwrap();
    loaded.validate().unwrap();
    assert_eq!(loaded, restored);
}

#[test]
fn capture_reuse_preserves_source_recipe_and_content_and_protects_shared_provenance() {
    use super::super::captures::{self, Action};
    let original = document();
    let change = captures::apply(
        &original,
        capture_request(&original, 2, "t0", Action::Attach),
    )
    .unwrap();
    let id = change.attached_id.unwrap();
    let repeated = captures::apply(
        &change.document,
        capture_request(&change.document, 2, "t0", Action::Attach),
    )
    .unwrap();
    assert_eq!(repeated.attached_id.as_ref(), Some(&id));
    assert_eq!(repeated.document.takes.len(), change.document.takes.len());
    let detached = captures::apply(
        &repeated.document,
        capture_request(&repeated.document, 2, &id.0, Action::Detach),
    )
    .unwrap();
    let restored = captures::apply(
        &detached.document,
        capture_request(&detached.document, 2, "t0", Action::Attach),
    )
    .unwrap();
    assert_eq!(restored.attached_id.as_ref(), Some(&id));
    assert_eq!(restored.document.takes.len(), change.document.takes.len());
    let copied = change.document.takes.iter().find(|t| t.id == id).unwrap();
    assert_eq!(copied.provenance, original.takes[0].provenance);
    assert_eq!(copied.scene_id.0, "s2");
    assert_eq!(copied.status, TakeStatus::Candidate);
    assert_eq!(
        change.document.captures[&id]
            .source_capture
            .as_ref()
            .unwrap()
            .0,
        "t0"
    );
    let new_media = change
        .document
        .resources
        .iter()
        .find(|r| matches!(&r.kind, ResourceKind::TakeVideo { take_id } if take_id == &id))
        .unwrap();
    assert_eq!(new_media.key(), original.resources[0].key());
    let removed = captures::apply(
        &change.document,
        capture_request(&change.document, 0, "t0", Action::Delete),
    )
    .unwrap();
    assert!(removed.retained_for_provenance);
    assert!(removed.document.captures[&TakeId("t0".into())].removed);
    assert!(removed.document.takes.iter().any(|t| t.id == id));
    assert_eq!(removed.document.resources, change.document.resources);
    removed.document.validate().unwrap();
}

#[test]
fn capture_delete_prunes_only_owned_take_descriptors_and_keeps_original_source() {
    use super::super::captures::{self, Action};
    let original = document();
    let change = captures::apply(
        &original,
        capture_request(&original, 2, "t2", Action::Delete),
    )
    .unwrap();
    assert!(!change.retained_for_provenance);
    assert!(!change.document.takes.iter().any(|t| t.id.0 == "t2"));
    assert_eq!(change.document.resources, original.resources[..2]);
    assert_eq!(original.takes.len(), 3);
    assert!(original.project.scenes[2].accepted_take.is_some());
    change.document.validate().unwrap();
}

#[test]
fn capture_conflict_and_cross_scene_removal_do_not_mutate_the_original() {
    use super::super::captures::{self, Action};
    let original = document();
    let bytes = serde_json::to_vec(&original).unwrap();
    let mut stale = capture_request(&original, 0, "t0", Action::Delete);
    stale.expected_project_revision -= 1;
    assert_eq!(
        captures::apply(&original, stale).err().unwrap().code,
        "revision_conflict"
    );
    assert_eq!(
        captures::apply(
            &original,
            capture_request(&original, 1, "t0", Action::Detach)
        )
        .err()
        .unwrap()
        .code,
        "take_scene_mismatch"
    );
    assert_eq!(serde_json::to_vec(&original).unwrap(), bytes);
}

#[test]
fn capture_catalog_can_remove_an_archived_scene_result_without_changing_other_scenes() {
    use super::super::captures::{self, Action};
    let mut original = document();
    original.project.scenes.pop();
    original.validate().unwrap();
    let changed = captures::apply(
        &original,
        capture_request(&original, 0, "t2", Action::Delete),
    )
    .unwrap();
    assert_eq!(changed.document.project.scenes, original.project.scenes);
    assert_eq!(changed.document.takes.len(), 2);
    changed.document.validate().unwrap();
}
