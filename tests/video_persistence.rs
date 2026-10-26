//! Only synthetic documents and temporary roots; no runtime files or models.
use imagine_server::video::{contract::*, editing::*, library::*, persistence::*};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Write},
    os::unix::fs::{PermissionsExt, symlink},
};

fn project() -> VideoProject {
    VideoProject::create(
        serde_json::from_str(include_str!("../design/video-contract/create-project.json")).unwrap(),
    )
    .unwrap()
}
fn input(id: &str, kind: ResourceKind, bytes: &[u8]) -> (Resource, File) {
    let mut f = tempfile::tempfile().unwrap();
    f.write_all(bytes).unwrap();
    let r = Resource::inspect(id.into(), kind, &mut f).unwrap();
    (r, f)
}
fn with_reference(staging: &SessionStaging) -> ProjectSession {
    let mut p = ProjectSession::new(project()).unwrap();
    let mut doc = p.document().clone();
    doc.project.assets.insert(
        AssetId("image".into()),
        Asset {
            library_source: None,
            name: String::new(),
            id: AssetId("image".into()),
            revision: 1,
            kind: MediaKind::Image,
        },
    );
    let (r, f) = input(
        "image",
        ResourceKind::Reference {
            asset_id: AssetId("image".into()),
            revision: 1,
        },
        b"SYNTHETIC IMAGE",
    );
    let key = r.key();
    doc.resources.push(r);
    p.update(0, doc, BTreeMap::from([(key, f)]), staging)
        .unwrap();
    p
}
fn prompt(p: &mut ProjectSession, text: &str) -> Result<(), StorageError> {
    p.edit(EditRequest {
        schema_version: 1,
        expected_project_revision: p.document().project.project_revision,
        commands: vec![Edit::SetGlobalPrompt {
            prompt: text.into(),
        }],
    })
}
fn bytes(p: &ProjectSession, id: &str) -> Vec<u8> {
    let mut result = vec![];
    p.copy_resource(id, &mut result).unwrap();
    result
}

#[test]
fn unsaved_project_has_only_unlinked_session_resources() {
    let root = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(root.path()).unwrap();
    let p = with_reference(&staging);
    assert_eq!(p.status().state, SaveState::SessionOnly);
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    assert_eq!(bytes(&p, "image"), b"SYNTHETIC IMAGE");
    drop(p);
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}
#[test]
fn save_open_and_autosave_own_references_after_session_deletion() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let output = tempfile::tempdir().unwrap();
    let mut p = with_reference(&staging);
    let id = p.document().project.id.clone();
    p.save_to(output.path()).unwrap();
    prompt(&mut p, "SYNTHETIC SAVED CHANGE").unwrap();
    let expected = p.document().clone();
    assert_eq!(p.status().state, SaveState::Saved);
    assert_eq!(p.status().saved_revision, Some(2));
    drop(p);
    drop(staging);
    drop(stage);
    let reopened = ProjectSession::open(output.path()).unwrap();
    assert_eq!(reopened.document(), &expected);
    assert_eq!(reopened.document().project.id, id);
    assert_eq!(bytes(&reopened, "image"), b"SYNTHETIC IMAGE");
    let manifest = fs::read_to_string(output.path().join("project.json")).unwrap();
    for key in ["job_id", "run_id", "idempotency", "session_id", "staging-"] {
        assert!(!manifest.contains(key));
    }
    assert_eq!(
        fs::metadata(output.path().join("project.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert_eq!(
        fs::metadata(output.path().join("media"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
}
#[test]
fn save_as_is_independent_and_failed_save_as_keeps_original_binding() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let original = tempfile::tempdir().unwrap();
    let copy = tempfile::tempdir().unwrap();
    let busy = tempfile::tempdir().unwrap();
    fs::write(busy.path().join("unrelated"), b"KEEP").unwrap();
    let mut p = with_reference(&staging);
    p.save_to(original.path()).unwrap();
    let old_id = p.document().project.id.clone();
    assert!(p.save_as(busy.path()).is_err());
    assert_eq!(p.document().project.id, old_id);
    prompt(&mut p, "ORIGINAL").unwrap();
    assert_eq!(
        ProjectSession::open(original.path())
            .unwrap()
            .document()
            .project
            .global_prompt,
        "ORIGINAL"
    );
    p.save_as(copy.path()).unwrap();
    assert_ne!(p.document().project.id, old_id);
    fs::remove_dir_all(original.path()).unwrap();
    prompt(&mut p, "COPY").unwrap();
    drop(p);
    let reopened = ProjectSession::open(copy.path()).unwrap();
    assert_eq!(bytes(&reopened, "image"), b"SYNTHETIC IMAGE");
    assert_eq!(reopened.document().project.global_prompt, "COPY");
    assert_eq!(fs::read(busy.path().join("unrelated")).unwrap(), b"KEEP");
}
#[test]
fn missing_or_replaced_root_retains_draft_and_never_creates_fallback() {
    let base = tempfile::tempdir().unwrap();
    let root = base.path().join("project");
    fs::create_dir(&root).unwrap();
    let mut p = ProjectSession::new(project()).unwrap();
    p.save_to(&root).unwrap();
    let before = fs::read(root.join("project.json")).unwrap();
    let moved = base.path().join("disconnected");
    fs::rename(&root, &moved).unwrap();
    fs::create_dir(&root).unwrap();
    assert!(prompt(&mut p, "UNSAVED").is_err());
    assert_eq!(p.document().project.global_prompt, "UNSAVED");
    assert_eq!(p.status().state, SaveState::NotSaved);
    assert_eq!(p.status().saved_revision, Some(0));
    assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
    assert_eq!(fs::read(moved.join("project.json")).unwrap(), before);
    fs::remove_dir(&root).unwrap();
    fs::rename(&moved, &root).unwrap();
    p.save().unwrap();
    assert_eq!(p.status().state, SaveState::Saved);
}
#[test]
fn stale_writers_cannot_replace_newer_manifest() {
    let root = tempfile::tempdir().unwrap();
    let mut first = ProjectSession::new(project()).unwrap();
    first.save_to(root.path()).unwrap();
    let mut stale = ProjectSession::open(root.path()).unwrap();
    prompt(&mut first, "FIRST").unwrap();
    assert_eq!(
        prompt(&mut stale, "STALE").unwrap_err().code,
        "storage_conflict"
    );
    assert_eq!(stale.document().project.global_prompt, "STALE");
    assert_eq!(stale.status().state, SaveState::NotSaved);
    assert_eq!(
        ProjectSession::open(root.path())
            .unwrap()
            .document()
            .project
            .global_prompt,
        "FIRST"
    );
}
#[test]
fn symlinks_and_tampered_resource_hashes_fail_closed() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let output = tempfile::tempdir().unwrap();
    let link = tempfile::tempdir().unwrap();
    symlink(output.path(), link.path().join("alias")).unwrap();
    assert!(
        ProjectSession::new(project())
            .unwrap()
            .save_to(&link.path().join("alias"))
            .is_err()
    );
    let mut p = with_reference(&staging);
    p.save_to(output.path()).unwrap();
    let hash = p.document().resources[0].sha256.clone();
    drop(p);
    fs::write(output.path().join("media").join(&hash), b"CHANGED").unwrap();
    assert!(ProjectSession::open(output.path()).is_err());
    fs::remove_file(output.path().join("media").join(&hash)).unwrap();
    let outside = link.path().join("original");
    fs::write(&outside, b"SYNTHETIC IMAGE").unwrap();
    symlink(&outside, output.path().join("media").join(hash)).unwrap();
    assert!(ProjectSession::open(output.path()).is_err());
    assert_eq!(fs::read(outside).unwrap(), b"SYNTHETIC IMAGE");
}
#[test]
fn invalid_documents_and_missing_inputs_do_not_mutate_session() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let mut p = ProjectSession::new(project()).unwrap();
    let before = p.document().clone();
    let mut invalid = before.clone();
    invalid.project.assets.insert(
        AssetId("missing".into()),
        Asset {
            library_source: None,
            name: String::new(),
            id: AssetId("missing".into()),
            revision: 0,
            kind: MediaKind::Video,
        },
    );
    assert!(p.update(0, invalid, BTreeMap::new(), &staging).is_err());
    assert_eq!(p.document(), &before);
    let (r, _) = input(
        "unavailable",
        ResourceKind::Reference {
            asset_id: AssetId("historical".into()),
            revision: 0,
        },
        b"NO SOURCE",
    );
    let mut missing = before.clone();
    missing.resources.push(r);
    assert_eq!(
        p.update(0, missing, BTreeMap::new(), &staging)
            .unwrap_err()
            .code,
        "resource_missing"
    );
    assert_eq!(p.document(), &before);
}
#[test]
fn library_project_copy_survives_source_and_library_deletion() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let library_root = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();
    let mut source = with_reference(&staging);
    let mut library = LibrarySession::create(library_root.path()).unwrap();
    let entry = library
        .save_project("Synthetic project".into(), None, &source)
        .unwrap();
    prompt(&mut source, "SOURCE CHANGED").unwrap();
    assert_eq!(library.document().revision, 1);
    let mut copied = library.copy_project(&entry, 1, &staging).unwrap();
    assert_ne!(copied.document().project.id, source.document().project.id);
    assert_eq!(copied.document().project.global_prompt, "");
    library
        .save_project("New revision".into(), Some((&entry, 1)), &source)
        .unwrap();
    assert!(library.copy_project(&entry, 1, &staging).is_err());
    assert_eq!(copied.document().origin.as_ref().unwrap().entry_revision, 1);
    drop(source);
    drop(library);
    fs::remove_dir_all(library_root.path()).unwrap();
    copied.save_to(output.path()).unwrap();
    drop(copied);
    assert_eq!(
        bytes(&ProjectSession::open(output.path()).unwrap(), "image"),
        b"SYNTHETIC IMAGE"
    );
}
#[test]
fn character_library_has_independent_media_and_voice_copy() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let (r, f) = input(
        "voice",
        ResourceKind::Reference {
            asset_id: AssetId("voice".into()),
            revision: 1,
        },
        b"SYNTHETIC AUDIO",
    );
    let key = r.key();
    let character = CharacterDocument {
        reference: None,
        schema_version: 1,
        id: "character".into(),
        name: "Synthetic".into(),
        alias: "synthetic".into(),
        description: "SYNTHETIC".into(),
        tags: vec!["test".into()],
        visual_ids: vec![],
        voice: Some(VoiceProfile {
            language: "sv".into(),
            description: "Synthetic voice".into(),
            reference_ids: vec!["voice".into()],
        }),
        resources: vec![r],
    };
    let mut library = LibrarySession::create(root.path()).unwrap();
    let entry = library
        .save_character(
            "Character".into(),
            None,
            character.clone(),
            BTreeMap::from([(key.clone(), f)]),
        )
        .unwrap();
    drop(library);
    let library = LibrarySession::open(root.path()).unwrap();
    let mut copy = library.copy_character(&entry, 1, &staging).unwrap();
    assert_eq!(copy.character, character);
    drop(library);
    fs::remove_dir_all(root.path()).unwrap();
    use std::io::{Seek, SeekFrom};
    let file = copy.sources.get_mut(&key).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut bytes = vec![];
    file.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"SYNTHETIC AUDIO");
}

#[test]
fn all_completed_take_statuses_and_historical_references_survive_open() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let output = tempfile::tempdir().unwrap();
    let mut p = ProjectSession::new(project()).unwrap();
    let mut doc = p.document().clone();
    let mut edits: EditRequest =
        serde_json::from_str(include_str!("../design/video-contract/edit-project.json")).unwrap();
    edits.expected_project_revision = 0;
    imagine_server::video::editing::apply(&mut doc.project, edits).unwrap();
    let model = doc.project.default_model.clone();
    let mut sources = BTreeMap::new();
    let (reference, file) = input(
        "old-reference",
        ResourceKind::Reference {
            asset_id: AssetId("historical".into()),
            revision: 3,
        },
        b"OLD REFERENCE",
    );
    sources.insert(reference.key(), file);
    doc.resources.push(reference);
    for (i, status) in [
        TakeStatus::Candidate,
        TakeStatus::Accepted,
        TakeStatus::Rejected,
        TakeStatus::Outdated,
    ]
    .into_iter()
    .enumerate()
    {
        let id = TakeId(format!("take-{i}"));
        let provenance = TakeProvenance {
            boundary: None,
            resolved_loras: None,
            model: model.clone(),
            seed: 42,
            input_revision: "synthetic-input".into(),
            compiled_prompt: "SYNTHETIC CAPTURED PROMPT".into(),
            settings: SettingsSlot {
                model: model.clone(),
                schema_version: 1,
                values: serde_json::json!({"synthetic":true}),
            },
            references: vec![AssetVersion {
                asset_id: AssetId("historical".into()),
                revision: 3,
            }],
            source_take: None,
            continuity: None,
            authored: None,
        };
        if status == TakeStatus::Accepted {
            doc.project.scenes[0].accepted_take = Some(AcceptedTake {
                id: id.clone(),
                model: Some(model.clone()),
                source_fingerprint: "synthetic-input".into(),
            });
        }
        doc.takes.push(StoredTake {
            id: id.clone(),
            scene_id: doc.project.scenes[0].id.clone(),
            status,
            provenance: Some(provenance),
        });
        let (r, f) = input(
            &format!("video-{i}"),
            ResourceKind::TakeVideo {
                take_id: id.clone(),
            },
            format!("SYNTHETIC VIDEO {i}").as_bytes(),
        );
        sources.insert(r.key(), f);
        doc.resources.push(r);
        let (r, f) = input(
            &format!("latent-{i}"),
            ResourceKind::Continuation {
                take_id: id,
                format: "synthetic-latent".into(),
                format_version: 7,
                model: model.clone(),
                width: 1280,
                height: 720,
            },
            b"OPAQUE SYNTHETIC LATENT",
        );
        sources.insert(r.key(), f);
        doc.resources.push(r);
    }
    p.update(0, doc, sources, &staging).unwrap();
    p.save_to(output.path()).unwrap();
    let expected = p.document().clone();
    drop(p);
    drop(staging);
    drop(stage);
    let mut reopened = ProjectSession::open(output.path()).unwrap();
    assert_eq!(reopened.document(), &expected);
    assert_eq!(reopened.document().takes.len(), 4);
    assert_eq!(bytes(&reopened, "old-reference"), b"OLD REFERENCE");
    assert_eq!(bytes(&reopened, "latent-3"), b"OPAQUE SYNTHETIC LATENT");
    let commands = reopened
        .document()
        .project
        .scenes
        .iter()
        .map(|scene| Edit::RemoveScene {
            scene_id: scene.id.clone(),
        })
        .collect();
    reopened
        .edit(EditRequest {
            schema_version: 1,
            expected_project_revision: reopened.document().project.project_revision,
            commands,
        })
        .unwrap();
    assert!(reopened.document().project.scenes.is_empty());
    assert_eq!(reopened.document().takes, expected.takes);
    assert_eq!(reopened.document().resources, expected.resources);
    drop(reopened);
    let retained = ProjectSession::open(output.path()).unwrap();
    assert_eq!(retained.document().takes, expected.takes);
    assert_eq!(bytes(&retained, "video-0"), b"SYNTHETIC VIDEO 0");
    assert_eq!(bytes(&retained, "latent-3"), b"OPAQUE SYNTHETIC LATENT");
}

#[test]
fn manifest_paths_and_unknown_versions_cannot_escape_root() {
    let root = tempfile::tempdir().unwrap();
    let mut p = ProjectSession::new(project()).unwrap();
    p.save_to(root.path()).unwrap();
    drop(p);
    let path = root.path().join("project.json");
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["schema_version"] = serde_json::json!(2);
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(ProjectSession::open(root.path()).is_err());
    value["schema_version"] = serde_json::json!(1);
    value["resources"] = serde_json::json!([{"id":"bad","kind":{"kind":"reference","asset_id":"a","revision":1},"sha256":"../../outside","bytes":1}]);
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(ProjectSession::open(root.path()).is_err());
}

#[test]
fn concurrent_resource_reads_do_not_share_seek_positions() {
    let stage = tempfile::tempdir().unwrap();
    let staging = SessionStaging::select(stage.path()).unwrap();
    let mut p = ProjectSession::new(project()).unwrap();
    let payload: Vec<u8> = (0..300_000).map(|i| (i % 251) as u8).collect();
    let (r, f) = input(
        "large",
        ResourceKind::Reference {
            asset_id: AssetId("large".into()),
            revision: 0,
        },
        &payload,
    );
    let mut doc = p.document().clone();
    let key = r.key();
    doc.resources.push(r);
    p.update(0, doc, BTreeMap::from([(key, f)]), &staging)
        .unwrap();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..4 {
                    assert_eq!(bytes(&p, "large"), payload);
                }
            });
        }
    });
}
