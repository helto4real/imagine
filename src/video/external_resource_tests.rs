//! Only new procedural source fixtures. No models, runtime data or user files.
use super::*;
use crate::video::library::{CharacterDocument, LibrarySession};
use std::{
    fs,
    io::Read,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
};

fn root() -> tempfile::TempDir {
    tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES")).unwrap()
}
fn project() -> VideoProject {
    VideoProject::create(
        serde_json::from_str(include_str!(
            "../../design/video-contract/create-project.json"
        ))
        .unwrap(),
    )
    .unwrap()
}
fn linked(root: &Path, kind: MediaKind) -> (ProjectSession, Resource, SessionStaging) {
    let original = root.join("own-original.bin");
    fs::write(&original, b"OWN PROCEDURAL SOURCE BYTES").unwrap();
    let mut file = File::open(&original).unwrap();
    let id = AssetId("own-external".into());
    let resource = Resource::external(
        "own-reference".into(),
        ResourceKind::Reference {
            asset_id: id.clone(),
            revision: 1,
        },
        &original,
        &mut file,
    )
    .unwrap();
    let mut session = ProjectSession::new(project()).unwrap();
    let mut document = session.document().clone();
    document.project.assets.insert(
        id.clone(),
        Asset {
            id,
            revision: 1,
            kind,
            name: "Own source fixture".into(),
            library_source: None,
        },
    );
    document.resources.push(resource.clone());
    let stage = root.join("stage");
    fs::create_dir(&stage).unwrap();
    let staging = SessionStaging::select(&stage).unwrap();
    session
        .update(
            0,
            document,
            BTreeMap::from([(resource.key(), file)]),
            &staging,
        )
        .unwrap();
    (session, resource, staging)
}
fn assert_original(file: &File, path: &Path) {
    let original = fs::metadata(path).unwrap();
    let actual = file.metadata().unwrap();
    assert_eq!(
        (actual.dev(), actual.ino()),
        (original.dev(), original.ino())
    );
    assert_eq!(
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) } & libc::O_ACCMODE,
        libc::O_RDONLY
    );
}

#[test]
fn external_project_save_library_and_copy_keep_original_without_blob_materialization() {
    let root = root();
    let (mut session, resource, staging) = linked(root.path(), MediaKind::Image);
    assert!(session.files().unwrap().is_empty());
    let saved = root.path().join("project");
    fs::create_dir(&saved).unwrap();
    session.save_to(&saved).unwrap();
    assert!(!saved.join("media").exists());
    let reopened = ProjectSession::open(&saved).unwrap();
    assert_original(
        &reopened.resource_file(&resource.id).unwrap(),
        resource.source.as_deref().unwrap(),
    );
    let library = root.path().join("library");
    fs::create_dir(&library).unwrap();
    let mut library = LibrarySession::create(&library).unwrap();
    let entry = library
        .save_project("Own project".into(), None, &reopened)
        .unwrap();
    let copied = library.copy_project(&entry, 1, &staging).unwrap();
    assert!(copied.files().unwrap().is_empty());
    assert_original(
        &copied.resource_file(&resource.id).unwrap(),
        resource.source.as_deref().unwrap(),
    );
    let asset = library
        .save_asset(
            "Own linked image".into(),
            None,
            &reopened,
            &AssetId("own-external".into()),
        )
        .unwrap();
    assert_original(
        &library.media_file(&asset, 1, &resource.id).unwrap().1,
        resource.source.as_deref().unwrap(),
    );
    assert!(!library.location().join("media").exists());
    assert_eq!(
        fs::read(resource.source.unwrap()).unwrap(),
        b"OWN PROCEDURAL SOURCE BYTES"
    );
}

#[test]
fn external_missing_or_changed_media_does_not_block_manifest_open_and_relink_metadata() {
    let root = root();
    let (mut session, mut resource, _) = linked(root.path(), MediaKind::Image);
    let original = resource.source.clone().unwrap();
    fs::write(&original, b"CHANGED OWN PROCEDURAL BYTES").unwrap();
    assert_eq!(
        session.resource_file(&resource.id).unwrap_err().code,
        "source_changed"
    );
    // An intentionally absent path, never delete or move an existing source.
    resource.source = Some(root.path().join("own-intentionally-absent.png"));
    session.document.resources[0] = resource.clone();
    let saved = root.path().join("project");
    fs::create_dir(&saved).unwrap();
    session.save_to(&saved).unwrap();
    let reopened = ProjectSession::open(&saved).unwrap();
    assert_eq!(
        reopened.resource_file(&resource.id).unwrap_err().code,
        "source_missing"
    );
    assert_eq!(fs::read(original).unwrap(), b"CHANGED OWN PROCEDURAL BYTES");
}

#[test]
fn external_character_copy_returns_original_descriptor_and_retains_path() {
    let root = root();
    let (session, resource, staging) = linked(root.path(), MediaKind::Image);
    let path = root.path().join("library");
    fs::create_dir(&path).unwrap();
    let mut library = LibrarySession::create(&path).unwrap();
    let character = CharacterDocument {
        schema_version: 1,
        id: "own-character".into(),
        name: "Own character".into(),
        alias: "own".into(),
        description: String::new(),
        tags: vec![],
        visual_ids: vec![resource.id.clone()],
        voice: None,
        resources: vec![resource.clone()],
        reference: None,
    };
    let entry = library
        .save_character(
            "Own character".into(),
            None,
            character,
            session.files().unwrap(),
        )
        .unwrap();
    let copy = library.copy_character(&entry, 1, &staging).unwrap();
    assert_eq!(copy.character.resources[0].source, resource.source);
    assert_original(
        &copy.sources[&resource.key()],
        resource.source.as_deref().unwrap(),
    );
    assert!(!path.join("media").exists());
}

#[test]
fn external_capture_inherits_link_and_delete_never_prunes_original() {
    use crate::video::captures::{Action, Import, Request};
    let root = root();
    let (mut session, resource, staging) = linked(root.path(), MediaKind::Video);
    let mut document = session.document().clone();
    let mut edit: EditRequest = serde_json::from_str(include_str!(
        "../../design/video-contract/edit-project.json"
    ))
    .unwrap();
    edit.expected_project_revision = document.project.project_revision;
    editing::apply(&mut document.project, edit).unwrap();
    // The synthetic edit changed only scene metadata; commit it through the normal boundary.
    let revision = session.document.project.project_revision;
    session
        .update(revision, document, BTreeMap::new(), &staging)
        .unwrap();
    let scene = session.document.project.scenes[0].id.clone();
    let saved = root.path().join("project");
    fs::create_dir(&saved).unwrap();
    session.save_to(&saved).unwrap();
    let change = session
        .import_take(
            Import {
                schema_version: 1,
                expected_project_revision: session.document.project.project_revision,
                scene_id: scene.clone(),
                asset_id: AssetId("own-external".into()),
                accept: false,
            },
            &staging,
        )
        .unwrap();
    let take = change.attached_id.unwrap();
    let imported = session
        .document
        .resources
        .iter()
        .find(|r| matches!(&r.kind,ResourceKind::TakeVideo{take_id} if take_id==&take))
        .unwrap();
    assert_eq!(imported.source, resource.source);
    assert_original(
        &session.resource_file(&imported.id).unwrap(),
        resource.source.as_deref().unwrap(),
    );
    let change = session
        .capture(
            Request {
                schema_version: 1,
                expected_project_revision: session.document.project.project_revision,
                scene_id: scene,
                take_id: take,
                action: Action::Delete,
            },
            &staging,
        )
        .unwrap();
    assert!(change.cleanup_warning.is_none());
    assert!(!saved.join("takes").exists());
    assert!(!saved.join("media").exists());
    assert_eq!(
        fs::read(resource.source.unwrap()).unwrap(),
        b"OWN PROCEDURAL SOURCE BYTES"
    );
}

#[test]
fn legacy_resource_json_omits_source_and_link_descriptor_has_independent_position() {
    let root = root();
    let path = root.path().join("own-bytes");
    fs::write(&path, b"OWN SOURCE").unwrap();
    let mut file = File::open(&path).unwrap();
    let legacy = Resource::inspect(
        "own".into(),
        ResourceKind::Reference {
            asset_id: AssetId("own".into()),
            revision: 1,
        },
        &mut file,
    )
    .unwrap();
    let json = serde_json::to_value(&legacy).unwrap();
    assert!(json.get("source").is_none());
    assert_eq!(serde_json::from_value::<Resource>(json).unwrap(), legacy);
    let external = Resource::external(legacy.id, legacy.kind, &path, &mut file).unwrap();
    let mut first = external.open_source().unwrap().unwrap();
    let mut second = external.open_source().unwrap().unwrap();
    let mut a = [0; 3];
    let mut b = [0; 3];
    first.read_exact(&mut a).unwrap();
    second.read_exact(&mut b).unwrap();
    assert_eq!(a, b);
}
