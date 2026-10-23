//! Source-owned synthetic project storage, never a user's project directory.
use super::*;
use crate::{private_storage::Root, video_destination as destination};
use std::{
    io::Read,
    sync::{Arc, Mutex},
};

struct Environment(Mutex<String>);
impl destination::Environment for Environment {
    fn mounts(&self) -> destination::Result<String> {
        Ok(self.0.lock().unwrap().clone())
    }
    fn host(&self) -> destination::Result<String> {
        Ok("own-private-project".into())
    }
}
fn fixture_root() -> (tempfile::TempDir, Root, Arc<Environment>) {
    let directory =
        tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES")).unwrap();
    let mount = directory.path().join("own-volume");
    let path = mount.join("imagine");
    std::fs::create_dir_all(&path).unwrap();
    let config = destination::Config {
        root: path,
        mount: mount.clone(),
        source: "own-volume".into(),
        filesystem: "fixturefs".into(),
        fsid: destination::fsid(&destination::directory(&mount).unwrap()).unwrap(),
    };
    let environment = Arc::new(Environment(Mutex::new(format!(
        "1 0 0:1 / {} rw - fixturefs own-volume rw\n",
        mount.display()
    ))));
    let runtime = directory.path().join("own-runtime");
    std::fs::create_dir(&runtime).unwrap();
    let root = Root::with_environment(config, environment.clone())
        .unwrap()
        .with_runtime_directory(runtime)
        .unwrap();
    (directory, root, environment)
}
fn session(path: &Path) -> ProjectSession {
    let project = VideoProject::create(
        serde_json::from_str(include_str!(
            "../../design/video-contract/create-project.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let mut session = ProjectSession::new(project).unwrap();
    let mut document = session.document().clone();
    let mut inputs = BTreeMap::new();
    for (name, kind, bytes) in [
        ("own-image", MediaKind::Image, b"OWN IMAGE".as_slice()),
        ("own-video", MediaKind::Video, b"OWN VIDEO".as_slice()),
        ("own-audio", MediaKind::Audio, b"OWN AUDIO".as_slice()),
    ] {
        let asset_id = AssetId(name.into());
        document.project.assets.insert(
            asset_id.clone(),
            Asset {
                id: asset_id.clone(),
                revision: 1,
                kind,
                name: String::new(),
                library_source: None,
            },
        );
        let source = path.join(name);
        std::fs::write(&source, bytes).unwrap();
        let mut file = File::open(source).unwrap();
        let resource = Resource::inspect(
            name.into(),
            ResourceKind::Reference {
                asset_id,
                revision: 1,
            },
            &mut file,
        )
        .unwrap();
        inputs.insert(resource.key(), file);
        document.resources.push(resource);
    }
    let original = path.join("own-original-reference");
    std::fs::write(&original, b"OWN LINK ONLY").unwrap();
    let asset_id = AssetId("own-link".into());
    document.project.assets.insert(
        asset_id.clone(),
        Asset {
            id: asset_id.clone(),
            revision: 1,
            kind: MediaKind::Image,
            name: String::new(),
            library_source: None,
        },
    );
    document.resources.push(
        Resource::external(
            "own-link".into(),
            ResourceKind::Reference {
                asset_id,
                revision: 1,
            },
            &original,
            &mut File::open(&original).unwrap(),
        )
        .unwrap(),
    );
    session
        .update(0, document, inputs, &SessionStaging::select(path).unwrap())
        .unwrap();
    session
}

#[test]
fn explicit_private_save_uses_output_classes_and_reopens_without_reference_copy() {
    let (fixtures, root, _) = fixture_root();
    let mut session = session(fixtures.path());
    assert!(session.document.storage_layout.is_none());
    session.save_private(&root).unwrap();
    assert_eq!(session.document.storage_layout, Some(1));
    let layout = root
        .layout()
        .project(&session.document.project.id.0)
        .unwrap();
    for resource in &session.document.resources {
        let directory = session.document.resource_directory(resource);
        assert_eq!(
            layout.root.join(directory).join(&resource.sha256).exists(),
            resource.source.is_none()
        );
    }
    assert!(!layout.root.join("media").exists());
    assert!(!layout.root.join("takes").exists());
    assert!(!layout.root.join("continuity").exists());
    assert!(!layout.root.join(".imagine-lock").exists());
    let locks = std::fs::read_dir(root.runtime_directory().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(locks.len(), 1);
    assert!(locks[0].starts_with("bundle-") && locks[0].ends_with(".lock"));
    assert!(!locks[0].contains(&session.document.project.id.0));
    for path in [
        &layout.images,
        &layout.videos,
        &layout.audio,
        &layout.work,
        &layout.tmp,
        &layout.autosave,
    ] {
        assert!(path.is_dir());
    }
    let reopened = ProjectSession::open_private(&root, &session.document.project.id).unwrap();
    assert_eq!(reopened.document, session.document);
    let mut bytes = vec![];
    reopened
        .resource_file("own-audio")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes, b"OWN AUDIO");
    let mut original = vec![];
    reopened
        .resource_file("own-link")
        .unwrap()
        .read_to_end(&mut original)
        .unwrap();
    assert_eq!(original, b"OWN LINK ONLY");
    let kind = ResourceKind::TakeVideo {
        take_id: TakeId("own-take".into()),
    };
    let mut resource = session.document.resources[0].clone();
    resource.kind = kind;
    assert_eq!(
        session.document.resource_directory(&resource),
        "outputs/videos"
    );
    resource.kind = ResourceKind::TakeAudio {
        take_id: TakeId("own-take".into()),
    };
    assert_eq!(
        session.document.resource_directory(&resource),
        "outputs/audio"
    );
    resource.kind = ResourceKind::Continuation {
        take_id: TakeId("own-take".into()),
        format: "own".into(),
        format_version: 1,
        model: session.document.project.default_model.clone(),
        width: 16,
        height: 16,
    };
    assert_eq!(
        session.document.resource_directory(&resource),
        "work/continuity"
    );
}

#[test]
fn private_first_save_failure_keeps_layout_and_sources_retryable() {
    let (fixtures, root, _) = fixture_root();
    let mut session = session(fixtures.path());
    let original = session.document.clone();
    let sources = std::mem::take(&mut session.sources);
    assert_eq!(
        session.save_private(&root).unwrap_err().code,
        "resource_missing"
    );
    assert_eq!(session.document, original);
    assert!(session.store.is_none());
    assert_eq!(session.status.state, SaveState::NotSaved);
    let layout = root.layout().project(&original.project.id.0).unwrap();
    assert!(!layout.document.exists());
    session.sources = sources;
    session.save_private(&root).unwrap();
    assert_eq!(session.document.storage_layout, Some(1));
    assert_eq!(session.status.state, SaveState::Saved);
}

#[test]
fn private_first_save_never_overwrites_manifest_and_legacy_stays_legacy() {
    let (fixtures, root, _) = fixture_root();
    let mut session = session(fixtures.path());
    let layout = root.project(&session.document.project.id.0).unwrap();
    std::fs::write(&layout.document, b"OWN EXISTING MANIFEST").unwrap();
    let document = session.document.clone();
    assert_eq!(
        session.save_private(&root).unwrap_err().code,
        "storage_not_empty"
    );
    assert_eq!(session.document, document);
    assert_eq!(
        std::fs::read(&layout.document).unwrap(),
        b"OWN EXISTING MANIFEST"
    );
    let legacy_folder = fixtures.path().join("own-legacy");
    std::fs::create_dir(&legacy_folder).unwrap();
    session.save_to(&legacy_folder).unwrap();
    let before = std::fs::read(legacy_folder.join("project.json")).unwrap();
    assert!(
        serde_json::from_slice::<serde_json::Value>(&before)
            .unwrap()
            .get("storage_layout")
            .is_none()
    );
    assert!(legacy_folder.join("media").is_dir());
    assert_eq!(
        session.save_private(&root).unwrap_err().code,
        "project_already_bound"
    );
    let reopened = ProjectSession::open(&legacy_folder).unwrap();
    assert!(reopened.document.storage_layout.is_none());
    assert_eq!(
        std::fs::read(legacy_folder.join("project.json")).unwrap(),
        before
    );
    assert!(!legacy_folder.join("outputs").exists());
}

#[test]
fn private_save_revalidates_volume_after_reopen_without_fallback() {
    let (fixtures, root, environment) = fixture_root();
    let mut session = session(fixtures.path());
    session.save_private(&root).unwrap();
    let layout = root
        .layout()
        .project(&session.document.project.id.0)
        .unwrap();
    let before = std::fs::read(&layout.document).unwrap();
    let mut reopened = ProjectSession::open_private(&root, &session.document.project.id).unwrap();
    environment.0.lock().unwrap().clear();
    assert_eq!(
        reopened.save().unwrap_err().code,
        "video_volume_not_mounted"
    );
    assert_eq!(reopened.document, session.document);
    assert_eq!(std::fs::read(&layout.document).unwrap(), before);
}

#[test]
fn explicit_legacy_open_is_read_only_and_save_as_creates_separate_private_project() {
    let (fixtures, root, _) = fixture_root();
    let mut original = session(fixtures.path());
    let folder = fixtures.path().join("own-legacy-read-only");
    std::fs::create_dir(&folder).unwrap();
    original.save_to(&folder).unwrap();
    let before = std::fs::read(folder.join("project.json")).unwrap();
    let mut legacy = ProjectSession::open_for_private(&root, &folder).unwrap();
    let unchanged = legacy.document.clone();
    assert_eq!(
        legacy.status.error_code.as_deref(),
        Some("legacy_project_requires_private_copy")
    );
    assert_eq!(
        legacy
            .edit(EditRequest {
                schema_version: 1,
                expected_project_revision: unchanged.project.project_revision,
                commands: vec![]
            })
            .unwrap_err()
            .code,
        "legacy_project_requires_private_copy"
    );
    assert_eq!(legacy.document, unchanged);
    assert_eq!(
        legacy.save().unwrap_err().code,
        "legacy_project_requires_private_copy"
    );
    let old_id = legacy.document.project.id.clone();
    let mut service = super::super::service::Service::default();
    service.projects.insert(legacy).unwrap();
    let copy = service
        .copy_private(
            &old_id,
            &root,
            &SessionStaging::select(fixtures.path()).unwrap(),
        )
        .unwrap();
    assert_ne!(copy, old_id);
    let copied = service.projects.session(&copy).unwrap();
    assert_eq!(copied.document.storage_layout, Some(1));
    assert_eq!(
        copied
            .document
            .resources
            .iter()
            .find(|r| r.id == "own-link")
            .unwrap()
            .source,
        original
            .document
            .resources
            .iter()
            .find(|r| r.id == "own-link")
            .unwrap()
            .source
    );
    assert_eq!(std::fs::read(folder.join("project.json")).unwrap(), before);
    assert!(root.layout().project(&copy.0).unwrap().document.exists());
    assert!(!folder.join("outputs").exists());
}

#[test]
fn private_library_metadata_stays_in_fixed_root_and_origin_is_not_copied() {
    let (fixtures, root, environment) = fixture_root();
    let project = session(fixtures.path());
    let mut library = super::super::library::LibrarySession::create_private(&root).unwrap();
    library
        .save_asset(
            "own-linked-entry".into(),
            None,
            &project,
            &AssetId("own-link".into()),
        )
        .unwrap();
    let location = root.layout().library();
    assert_eq!(library.location(), location);
    assert!(location.join("library.json").exists());
    assert!(!location.join("media").exists());
    assert!(!location.join(".imagine-lock").exists());
    let document = library.document().clone();
    let mut reopened = super::super::library::LibrarySession::open_private(&root).unwrap();
    assert_eq!(reopened.document(), &document);
    environment.0.lock().unwrap().clear();
    assert_eq!(
        reopened
            .edit_entry(
                &document.entries[0].id,
                document.entries[0].revision,
                Some("own changed title".into()),
                None,
                false
            )
            .unwrap_err()
            .code,
        "video_volume_not_mounted"
    );
    assert_eq!(reopened.document(), &document);
}

#[test]
fn explicit_private_older_export_preserves_active_layout_and_uses_legacy_copy_paths() {
    let (fixtures, root, _) = fixture_root();
    let mut session = session(fixtures.path());
    // The older format cannot represent original links. This own fixture keeps
    // only the representable owned resources, without copying any external file.
    let mut document = session.document.clone();
    document
        .resources
        .retain(|resource| resource.source.is_none());
    document.project.assets.remove(&AssetId("own-link".into()));
    let expected_revision = document.project.project_revision;
    session
        .update(
            expected_revision,
            document,
            BTreeMap::new(),
            &SessionStaging::select(fixtures.path()).unwrap(),
        )
        .unwrap();
    session.save_private(&root).unwrap();
    let before = session.document.clone();
    let report = session.export_compatible_private(&root).unwrap();
    assert!(report.retained_private_layout_in_original);
    assert_ne!(report.project_id, before.project.id);
    assert_eq!(session.document, before);
    let layout = root.layout().project(&report.project_id.0).unwrap();
    let copy = ProjectSession::open_for_private(&root, &layout.root).unwrap();
    assert!(copy.document.storage_layout.is_none());
    for resource in &copy.document.resources {
        assert!(layout.root.join("media").join(&resource.sha256).is_file());
    }
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&layout.document).unwrap()).unwrap();
    assert!(json.get("storage_layout").is_none());
    assert!(!layout.root.join(".imagine-lock").exists());
}

#[test]
fn private_library_create_preserves_only_the_decided_shared_metadata_files() {
    let (_fixtures, root, _) = fixture_root();
    let library = root.library().unwrap();
    let names = [
        "image-workspace.json",
        "enhancement-profiles.json",
        "ui-preferences.json",
    ];
    for name in names {
        std::fs::write(library.join(name), b"OWN PROCEDURAL METADATA STUB").unwrap();
    }
    let catalog = library.join("prompt-catalog/catalog-own");
    std::fs::create_dir_all(&catalog).unwrap();
    std::fs::write(catalog.join("own.yaml"), b"OWN UNSUBMITTED CATALOG").unwrap();
    let created = super::super::library::LibrarySession::create_private(&root).unwrap();
    assert_eq!(created.location(), library);
    assert!(library.join("library.json").is_file());
    assert_eq!(
        std::fs::read(catalog.join("own.yaml")).unwrap(),
        b"OWN UNSUBMITTED CATALOG"
    );
    for name in names {
        assert_eq!(
            std::fs::read(library.join(name)).unwrap(),
            b"OWN PROCEDURAL METADATA STUB"
        );
    }

    let (_other_fixtures, other_root, _) = fixture_root();
    let other = other_root.library().unwrap();
    std::fs::create_dir(other.join("ui-preferences.json")).unwrap();
    let error = super::super::library::LibrarySession::create_private(&other_root)
        .err()
        .unwrap();
    assert_eq!(error.code, "storage_not_empty");
    assert!(!other.join("library.json").exists());
}
