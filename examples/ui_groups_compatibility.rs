//! Opt-in, media-free CPU acceptance against an actual frozen older ProjectSession.
use imagine_server::video::{
    contract::*,
    editing::{Edit, EditRequest},
    persistence::ProjectSession,
};
use imagine_video_contract::assembly::{Boundary, BoundaryMode};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};

#[path = "../shared/task_fixtures.rs"]
mod task_fixtures;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_legacy(reader: &Path, folder: &Path) -> Value {
    let result = Command::new(reader)
        .arg(folder)
        .output()
        .expect("run isolated baseline reader");
    assert!(result.status.success(), "baseline reader process failed");
    serde_json::from_slice(&result.stdout).expect("structured baseline reader result")
}
fn time(num: u64) -> Time {
    Time { num, den: 1 }
}
fn main() {
    let root = task_fixtures::env_root("IMAGINE_COMPAT_FIXTURE_ROOT");
    assert!(
        root.read_dir().unwrap().next().is_none(),
        "new own fixture directory only"
    );
    let reader = std::path::PathBuf::from(
        std::env::var_os("IMAGINE_LEGACY_READER").expect("explicit baseline reader binary"),
    );
    assert!(reader.is_absolute() && reader.is_file());
    let reader_sha256 = digest(&fs::read(&reader).unwrap());
    let defaults = imagine_video_contract::generation::Defaults::default();
    let mut project = VideoProject::create(CreateProject {
        generation: Some(defaults),
        schema_version: SCHEMA_VERSION,
        default_model: imagine_video_contract::generation::h3_model("reference"),
        delivery: DeliverySettings {
            width: 416,
            height: 224,
            fps: time(24),
        },
    })
    .unwrap();
    for (id, start) in [("own-left", 0), ("own-right", 1)] {
        project.scenes.push(Scene {
            id: SceneId(id.into()),
            start: time(start),
            duration: time(1),
            description: "Synthetic compatibility fixture".into(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
    }
    let boundary = Boundary {
        mode: BoundaryMode::BlendSeam,
        ..Boundary::new(SceneId("own-left".into()), SceneId("own-right".into()))
    };
    project.assembly.boundaries.push(boundary.clone());
    let mut session = ProjectSession::new(project).unwrap();
    let original = root.join("original");
    let compatible = root.join("compatible-defaults");
    let rejected = root.join("rejected-authored-boundary");
    for folder in [&original, &compatible, &rejected] {
        fs::create_dir(folder).unwrap();
    }
    session.save_to(&original).unwrap();
    let initial_document = session.document().clone();
    let initial_status = session.status().clone();
    let initial_bytes = fs::read(original.join("project.json")).unwrap();
    let default_json: Value = serde_json::from_slice(&initial_bytes).unwrap();
    let default_boundary = default_json
        .pointer("/project/assembly/boundaries/0")
        .unwrap();
    for field in [
        "tail_frames",
        "transition_prompt",
        "reuse_visual_references",
        "reuse_style",
        "audio_continuity",
    ] {
        assert!(
            default_boundary.get(field).is_none(),
            "default field must remain sparse: {field}"
        );
    }
    let report = session.export_compatible(&compatible).unwrap();
    assert_eq!(session.document(), &initial_document);
    assert_eq!(session.status(), &initial_status);
    assert_eq!(
        fs::read(original.join("project.json")).unwrap(),
        initial_bytes
    );
    let old = read_legacy(&reader, &compatible);
    assert_eq!(
        old["ok"], true,
        "actual older reader must open the explicit copy: {old}"
    );
    assert_eq!(old["takes"], 0);
    assert_eq!(old["resources"], 0);
    assert_eq!(
        old["boundaries"][0],
        serde_json::to_value(&boundary).unwrap()
    );
    assert_eq!(old["scenes"].as_array().unwrap().len(), 2);
    let compatible_session = ProjectSession::open(&compatible).unwrap();
    assert_ne!(
        compatible_session.document().project.id,
        session.document().project.id
    );
    assert!(compatible_session.document().project.generation.is_none());
    for scene in &session.document().project.scenes {
        let copy = compatible_session
            .document()
            .project
            .scene(&scene.id)
            .unwrap();
        assert_eq!(
            session.document().project.effective_settings(scene),
            compatible_session
                .document()
                .project
                .effective_settings(copy)
        );
    }
    let mut authored = boundary;
    authored.mode = BoundaryMode::Transition;
    authored.tail_frames = Some(23);
    authored.transition_prompt = "Synthetic authored boundary transition".into();
    session
        .edit(EditRequest {
            schema_version: SCHEMA_VERSION,
            expected_project_revision: session.document().project.project_revision,
            commands: vec![Edit::SetAssemblyBoundary {
                boundary: authored.clone(),
            }],
        })
        .unwrap();
    let authored_document = session.document().clone();
    let authored_status = session.status().clone();
    let authored_bytes = fs::read(original.join("project.json")).unwrap();
    let reopened = ProjectSession::open(&original).unwrap();
    assert_eq!(reopened.document(), &authored_document);
    assert_eq!(reopened.document().project.assembly.boundaries, [authored]);
    let error = session.export_compatible(&rejected).unwrap_err();
    assert_eq!(error.code, "older_format_boundary_not_representable");
    assert_eq!(session.document(), &authored_document);
    assert_eq!(session.status(), &authored_status);
    assert_eq!(
        fs::read(original.join("project.json")).unwrap(),
        authored_bytes
    );
    assert!(rejected.read_dir().unwrap().next().is_none());
    assert_eq!(digest(&fs::read(&reader).unwrap()), reader_sha256);
    let proof = json!({
        "schema_version":1,"ok":true,"workload":"synthetic-media-free-project",
        "original":original,"compatible_copy":compatible,"rejected_copy":rejected,
        "legacy_reader":reader,"legacy_reader_sha256":reader_sha256,"legacy_result":old,
        "compatibility_report":report,"initial_original_sha256":digest(&initial_bytes),
        "authored_original_sha256":digest(&authored_bytes),
        "authored_tail_frames":23,"authored_transition_prompt_retained":true,
        "refused_export_code":error.code,"refused_export_left_target_empty":true,
        "original_document_status_and_disk_unchanged_by_export":true,
        "limitations":["No media, Takes, library, or Story/provider formats exercised", "No server, UI, inference, or GPU"]
    });
    fs::write(
        root.join("proof.json"),
        serde_json::to_vec_pretty(&proof).unwrap(),
    )
    .unwrap();
    println!("{proof}");
}
