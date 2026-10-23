//! CPU montage acceptance with procedural frames/tones on the own encrypted test volume.
#[path = "../ui/tests/support/destination.rs"]
#[allow(dead_code)]
mod destination;
use imagine_server::{
    engine::{Backend, EngineFuture},
    scheduler::{self, Action},
    short_av,
    storage::Store,
    video::{assembly::*, contract::*, persistence::*, service::Service},
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::Write,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};
fn own() -> tempfile::TempDir {
    let root = std::env::var("TMPDIR").unwrap();
    assert!(root.starts_with("/run/media/veracrypt1/imagine/.codex-"));
    tempfile::tempdir_in(root).unwrap()
}
fn fixture(root: &Path, mode: BoundaryMode) -> (Service, Request, Vec<u8>) {
    let spec = short_av::EncodeSpec {
        width: 64,
        height: 64,
        frames: 24,
        fps_num: 24,
        fps_den: 1,
        sample_rate: 44100,
        bit_depth: 8,
    };
    let pcm = (0..44100)
        .flat_map(|i| {
            [
                0.1 * (std::f32::consts::TAU * 440. * i as f32 / 44100.).sin(),
                0.08 * (std::f32::consts::TAU * 660. * i as f32 / 44100.).sin(),
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    let bytes = short_av::encode_stream(
        spec,
        (0..24).map(|i| Ok(vec![40 + i * 5; 64 * 64 * 3])),
        pcm,
        &AtomicBool::new(false),
    )
    .unwrap();
    let model = ModelSelection {
        adapter: "any".into(),
        model: "unused".into(),
        variant: "unused".into(),
        profile: "unused".into(),
        profile_revision: "1".into(),
    };
    let mut project = VideoProject::create(CreateProject {
        schema_version: 1,
        default_model: model,
        delivery: DeliverySettings {
            width: 64,
            height: 64,
            fps: Time { num: 24, den: 1 },
        },
    })
    .unwrap();
    let asset = AssetId("owned_source".into());
    project.assets.insert(
        asset.clone(),
        Asset {
            id: asset.clone(),
            revision: 1,
            kind: MediaKind::Video,
        },
    );
    for i in 0..2 {
        let id = SceneId(format!("scene_{i}"));
        project.scenes.push(Scene {
            id: id.clone(),
            start: time(i as f64 * 2.),
            duration: time(1.),
            description: "own procedural fixture".into(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
        project.assembly.clips.insert(
            id,
            ClipInstance {
                asset: asset.clone(),
                source_in: time(i as f64 / 2.),
                source_out: Some(time((i + 1) as f64 / 2.)),
                speed: 0.5,
                include_audio: true,
            },
        );
    }
    project.assembly.boundaries.push(Boundary {
        left: SceneId("scene_0".into()),
        right: SceneId("scene_1".into()),
        mode,
        blend_frames: 3,
    });
    let mut source = tempfile::tempfile_in(root).unwrap();
    source.write_all(&bytes).unwrap();
    let resource = Resource::inspect(
        "source".into(),
        ResourceKind::Reference {
            asset_id: asset,
            revision: 1,
        },
        &mut source,
    )
    .unwrap();
    let mut doc = ProjectDocument::new(project.clone());
    doc.resources.push(resource.clone());
    let mut session = ProjectSession::new(project).unwrap();
    session
        .update(
            0,
            doc,
            BTreeMap::from([(resource.key(), source)]),
            &SessionStaging::select(root).unwrap(),
        )
        .unwrap();
    let id = session.document().project.id.clone();
    let revision = session.document().project.project_revision;
    let mut service = Service::default();
    service.projects.insert(session).unwrap();
    (
        service,
        Request {
            schema_version: 1,
            project_id: id,
            expected_project_revision: revision,
            options: Options {
                normalize_mix: false,
                ..Default::default()
            },
        },
        bytes,
    )
}
fn probe(root: &Path, bytes: &[u8]) -> Value {
    let mut file = tempfile::NamedTempFile::new_in(root).unwrap();
    file.write_all(bytes).unwrap();
    let result=std::process::Command::new("/usr/bin/ffprobe").args(["-v","error","-count_frames","-show_entries","format=duration:stream=codec_name,nb_read_frames,pix_fmt,sample_rate,channels,color_space,duration","-of","json"]).arg(file.path()).output().unwrap();
    assert!(result.status.success());
    serde_json::from_slice(&result.stdout).unwrap()
}
#[test]
fn imported_trim_blend_has_exact_frame_clock_stereo_audio_and_no_timeline_gap() {
    let root = own();
    let (service, request, _) = fixture(root.path(), BoundaryMode::BlendSeam);
    let frozen = freeze(
        &service,
        &request,
        &SessionStaging::select(root.path()).unwrap(),
    )
    .unwrap();
    assert_eq!(frozen.plan.frames, 45);
    assert_eq!(frozen.plan.clips[1].output_start, 21);
    assert_eq!(frozen.plan.samples, 82688);
    let bytes = render(frozen, Arc::default(), Progress::default()).unwrap();
    let value = probe(root.path(), &bytes);
    assert_eq!(value["streams"][0]["nb_read_frames"], "45");
    assert_eq!(value["streams"][0]["codec_name"], "h264");
    assert_eq!(value["streams"][0]["pix_fmt"], "yuv420p");
    assert_eq!(value["streams"][0]["color_space"], "bt709");
    assert_eq!(value["streams"][1]["codec_name"], "aac");
    assert_eq!(value["streams"][1]["sample_rate"], "44100");
    assert_eq!(value["streams"][1]["channels"], 2);
    assert!(
        (value["format"]["duration"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap()
            - 1.875)
            .abs()
            < 0.001
    );
    let mut file = tempfile::NamedTempFile::new_in(root.path()).unwrap();
    file.write_all(&bytes).unwrap();
    let pcm = std::process::Command::new("/usr/bin/ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(file.path())
        .args(["-vn", "-f", "f32le", "-ac", "2", "-ar", "44100", "pipe:1"])
        .output()
        .unwrap();
    assert!(pcm.status.success());
    let values = pcm
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let rms =
        (values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    assert!((0.03..0.11).contains(&rms), "{rms}");
}
#[test]
fn transition_is_explicit_and_cancelled_render_has_no_container() {
    let root = own();
    let (mut service, request, _) = fixture(root.path(), BoundaryMode::Transition);
    let staging = SessionStaging::select(root.path()).unwrap();
    assert_eq!(
        freeze(&service, &request, &staging).err().unwrap().code,
        "assembly_transition_take_required"
    );
    let p = service.projects.read(&request.project_id).unwrap();
    service
        .projects
        .edit(
            &p.id,
            imagine_server::video::editing::EditRequest {
                schema_version: 1,
                expected_project_revision: p.project_revision,
                commands: vec![imagine_server::video::editing::Edit::SetAssemblyBoundary {
                    boundary: Boundary {
                        left: SceneId("scene_0".into()),
                        right: SceneId("scene_1".into()),
                        mode: BoundaryMode::HardCut,
                        blend_frames: 0,
                    },
                }],
            },
        )
        .unwrap();
    let mut request = request;
    request.expected_project_revision += 1;
    let frozen = freeze(&service, &request, &staging).unwrap();
    assert_eq!(
        render(frozen, Arc::new(AtomicBool::new(true)), Progress::default()),
        Err("media_process_cancelled")
    );
}
struct NoGpu;
impl Backend for NoGpu {
    fn generate<'a>(
        &'a mut self,
        _: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        panic!("CPU assembly touched inference");
    }
    fn begin_job(&mut self, _: &str) {
        panic!("CPU assembly began GPU job");
    }
    fn mark_idle(&mut self) {
        panic!("CPU assembly changed GPU state");
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
async fn terminal(store: &Store, id: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let job = store.job(id, "local").await.unwrap().unwrap();
            if matches!(
                job["status"].as_str(),
                Some("succeeded" | "failed" | "cancelled")
            ) {
                return job;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
#[tokio::test]
async fn cpu_fifo_never_contacts_gpu_and_delivery_is_saved_once_and_reopens() {
    let root = own();
    let mut store = Store::open(root.path()).await.unwrap();
    store.videos.destination = Some(destination::destination(&root.path().join("output")));
    let (control, worker) = scheduler::start_coordinated(
        store.clone(),
        Box::new(NoGpu),
        Some(root.path().join("nonexistent-gpu-coordinator.sock")),
    )
    .await
    .unwrap();
    let (mut service, request, _) = fixture(root.path(), BoundaryMode::HardCut);
    let staging = SessionStaging::select(root.path()).unwrap();
    let folder = root.path().join("saved-project");
    std::fs::create_dir(&folder).unwrap();
    service
        .projects
        .session_mut(&request.project_id)
        .unwrap()
        .save_to(&folder)
        .unwrap();
    let frozen = freeze(&service, &request, &staging).unwrap();
    let plan_id = frozen.plan.id.clone();
    service.assemblies.insert(
        plan_id.clone(),
        Entry {
            frozen,
            progress: Default::default(),
            job: None,
            rendered: Default::default(),
        },
    );
    *control.video.lock().await = service;
    let submit = || Action::SubmitAssembly {
        owner: "local".into(),
        plan_id: plan_id.clone(),
        key: "one_export".into(),
    };
    let admitted = control.call(submit()).await.unwrap();
    let duplicate = control.call(submit()).await.unwrap();
    assert_eq!(admitted["id"], duplicate["id"]);
    assert_eq!(duplicate["reused"], true);
    let done = terminal(&store, admitted["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(done["output"]["video"]["frames"], 48);
    assert!(done["output"]["delivery"].is_object());
    assert!(done["output"].get("take").is_none());
    let project = ProjectSession::open(&folder).unwrap();
    assert_eq!(project.document().deliveries.len(), 1);
    assert!(project.document().takes.is_empty());
    assert_eq!(project.document().project.assembly.clips.len(), 2);
    let resource=project.document().resources.iter().find(|r|matches!(&r.kind,ResourceKind::Reference{asset_id,..}if asset_id==&project.document().deliveries[0].asset_id)).unwrap();
    assert!(
        project
            .resource_file(&resource.id)
            .unwrap()
            .metadata()
            .unwrap()
            .len()
            > 0
    );
    control.call(Action::Shutdown).await.unwrap();
    worker.await.unwrap().unwrap();
    store.close().await.unwrap();
}
#[tokio::test]
async fn queued_and_running_cancel_join_cpu_children_then_retry_and_failure_remain_separate() {
    let root = own();
    let mut store = Store::open(root.path()).await.unwrap();
    store.videos.destination = Some(destination::destination(&root.path().join("output")));
    let (control, worker) = scheduler::start(store.clone(), Box::new(NoGpu))
        .await
        .unwrap();
    let (mut service, request, _) = fixture(root.path(), BoundaryMode::HardCut);
    let staging = SessionStaging::select(root.path()).unwrap();
    let frozen = freeze(&service, &request, &staging).unwrap();
    let plan_id = frozen.plan.id.clone();
    service.assemblies.insert(
        plan_id.clone(),
        Entry {
            frozen,
            progress: Default::default(),
            job: None,
            rendered: Default::default(),
        },
    );
    *control.video.lock().await = service;
    let admit = |key: &str| Action::SubmitAssembly {
        owner: "local".into(),
        plan_id: plan_id.clone(),
        key: key.into(),
    };
    let first = control.call(admit("cancel_running")).await.unwrap();
    let first = first["id"].as_str().unwrap().to_owned();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let job = store.job(&first, "local").await.unwrap().unwrap();
            let phase = control.video.lock().await.assemblies[&plan_id]
                .progress
                .read()
                .phase;
            if job["status"] == "running"
                && matches!(phase.as_str(), "decoding_clip" | "encoding_video")
            {
                break;
            }
            assert_ne!(job["status"], "succeeded");
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let second = control.call(admit("cancel_queued")).await.unwrap();
    let second = second["id"].as_str().unwrap().to_owned();
    control
        .call(Action::Cancel {
            owner: "local".into(),
            id: second.clone(),
        })
        .await
        .unwrap();
    control
        .call(Action::Cancel {
            owner: "local".into(),
            id: first.clone(),
        })
        .await
        .unwrap();
    assert_eq!(terminal(&store, &first).await["status"], "cancelled");
    assert_eq!(terminal(&store, &second).await["status"], "cancelled");
    assert!(
        control
            .video
            .lock()
            .await
            .projects
            .session(&request.project_id)
            .unwrap()
            .document()
            .deliveries
            .is_empty()
    );
    let retry = control.call(admit("retry_after_cancel")).await.unwrap();
    assert_eq!(
        terminal(&store, retry["id"].as_str().unwrap()).await["status"],
        "succeeded"
    );
    let bad_id = {
        let mut service = control.video.lock().await;
        let mut bad = service.assemblies[&plan_id].frozen.try_clone().unwrap();
        bad.plan.id = "assembly_bad_range".into();
        bad.plan.clips[0].source_in = time(100.);
        bad.plan.clips[0].source_out = None;
        let id = bad.plan.id.clone();
        service.assemblies.insert(
            id.clone(),
            Entry {
                frozen: bad,
                progress: Default::default(),
                job: None,
                rendered: Default::default(),
            },
        );
        id
    };
    let bad = control
        .call(Action::SubmitAssembly {
            owner: "local".into(),
            plan_id: bad_id,
            key: "failed_range".into(),
        })
        .await
        .unwrap();
    let failed = terminal(&store, bad["id"].as_str().unwrap()).await;
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["code"], "assembly_empty_source_range");
    assert!(failed["output"].is_null());
    assert_eq!(
        control
            .video
            .lock()
            .await
            .projects
            .session(&request.project_id)
            .unwrap()
            .document()
            .deliveries
            .len(),
        1
    );
    control.call(Action::Shutdown).await.unwrap();
    worker.await.unwrap().unwrap();
    store.close().await.unwrap();
}

#[tokio::test]
async fn publication_failure_retains_encoded_result_and_retry_does_not_decode_or_render() {
    let root = own();
    let mut store = Store::open(root.path()).await.unwrap();
    let output = root.path().join("output");
    store.videos.destination = Some(destination::destination(&output));
    let (control, worker) = scheduler::start(store.clone(), Box::new(NoGpu))
        .await
        .unwrap();
    let (mut service, request, _) = fixture(root.path(), BoundaryMode::BlendSeam);
    let frozen = freeze(
        &service,
        &request,
        &SessionStaging::select(root.path()).unwrap(),
    )
    .unwrap();
    let plan_id = frozen.plan.id.clone();
    let retained = Rendered::default();
    service.assemblies.insert(
        plan_id.clone(),
        Entry {
            frozen,
            progress: Default::default(),
            job: None,
            rendered: retained.clone(),
        },
    );
    *control.video.lock().await = service;
    // Real publication failure after encoding and filesystem publication, before SQL commit.
    sqlx::query("CREATE TRIGGER own_fixture_save_failure BEFORE INSERT ON assets BEGIN SELECT RAISE(FAIL,'own fixture publication error'); END").execute(&store.db).await.unwrap();
    let submit = |key: &str| Action::SubmitAssembly {
        owner: "local".into(),
        plan_id: plan_id.clone(),
        key: key.into(),
    };
    let first = control.call(submit("publication_failure")).await.unwrap();
    let failed = terminal(&store, first["id"].as_str().unwrap()).await;
    assert_eq!(failed["status"], "failed", "{failed}");
    assert_eq!(failed["error"]["code"], "output_save_failed");
    assert!(failed["output"].is_null());
    let bytes = retained
        .get()
        .expect("completed MP4 must survive save failure");
    let expected_hash = imagine_server::storage::hash(&bytes);
    assert!(!bytes.is_empty());
    assert!(
        control
            .video
            .lock()
            .await
            .projects
            .session(&request.project_id)
            .unwrap()
            .document()
            .deliveries
            .is_empty()
    );
    assert!(
        std::fs::read_dir(&output)
            .unwrap()
            .flat_map(|d| std::fs::read_dir(d.unwrap().path()).unwrap())
            .next()
            .is_none(),
        "failed publication must roll back the output file"
    );
    let same = control.call(submit("publication_failure")).await.unwrap();
    assert_eq!(same["id"], first["id"]);
    assert_eq!(same["reused"], true);
    sqlx::query("DROP TRIGGER own_fixture_save_failure")
        .execute(&store.db)
        .await
        .unwrap();
    // A re-render now cannot work: the frozen decoder has no source descriptors.
    control
        .video
        .lock()
        .await
        .assemblies
        .get_mut(&plan_id)
        .unwrap()
        .frozen
        .files
        .clear();
    let retry = control.call(submit("save_retained_output")).await.unwrap();
    let done = terminal(&store, retry["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(done["output"]["assets"][0]["sha256"], expected_hash);
    assert!(
        retained.get().is_none(),
        "release the encoded cache after successful publication"
    );
    assert_eq!(
        control
            .video
            .lock()
            .await
            .projects
            .session(&request.project_id)
            .unwrap()
            .document()
            .deliveries
            .len(),
        1
    );
    control.call(Action::Shutdown).await.unwrap();
    worker.await.unwrap().unwrap();
    store.close().await.unwrap();
}
