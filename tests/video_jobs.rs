//! Synthetic scene jobs share the real FIFO, output transaction and private coordinator protocol.
#[path = "../ui/tests/support/destination.rs"]
#[allow(dead_code)]
mod destination;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use imagine_server::{
    api::{self, App},
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action, Control},
    storage::Store,
};
use serde_json::{Value, json};
use std::{
    io::Cursor,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{Notify, Semaphore, watch},
    task::JoinHandle,
};
use tower::ServiceExt;
#[derive(Default)]
struct Probe {
    calls: Mutex<Vec<String>>,
    references: Mutex<Vec<imagine_server::h3_protocol::references::Input>>,
    release: Notify,
    fail: AtomicBool,
    stops: AtomicUsize,
    produced: Notify,
}
struct Fake(Arc<Probe>);
impl Backend for Fake {
    fn generate<'a>(
        &'a mut self,
        p: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            self.0.calls.lock().unwrap().push(p.prompt.clone());
            self.0.release.notified().await;
            let mut b = Cursor::new(vec![]);
            image::RgbImage::new(2, 2).write_to(&mut b, image::ImageFormat::Png)?;
            Ok(b.into_inner())
        })
    }
    fn generate_video<'a>(
        &'a mut self,
        p: &'a imagine_server::h3_protocol::Request,
    ) -> EngineFuture<'a, imagine_server::short_av::Av> {
        Box::pin(async move {
            anyhow::ensure!(p.references.valid_payload());
            self.0.references.lock().unwrap().push(p.references.clone());
            if p.profile == imagine_server::h3_protocol::Profile::ImageToAvV1 {
                if p.prompt.contains("own continuation target") {
                    anyhow::ensure!(
                        p.valid()
                            && p.image_rgb.len() == 32 * 32 * 3
                            && p.image_rgb.iter().all(|pixel| *pixel == 70)
                    );
                } else {
                    anyhow::ensure!(p.valid() && p.image_rgb.len() == 16 * 16 * 3);
                    anyhow::ensure!(
                        p.image_rgb
                            .chunks_exact(3)
                            .all(|pixel| pixel == [50, 80, 120])
                    );
                }
            }
            self.0.calls.lock().unwrap().push(p.prompt.clone());
            self.0.release.notified().await;
            if self.0.fail.swap(false, Ordering::SeqCst) {
                use imagine_server::h3_protocol::{Failure, FailureKind, Stage};
                let failure = Failure {
                    stage: Some(Stage::PreparingMotionContext),
                    kind: Some(FailureKind::ShapeMismatch),
                    sizes: vec![7, 8],
                    ..Default::default()
                };
                return Err(
                    anyhow::Error::new(imagine_server::engine::H3Failure(failure))
                        .context(imagine_server::engine::EngineError("h3_generation_failed")),
                );
            }
            self.0.produced.notify_one();
            let continuation = p.capture_continuation.then(|| {
                let descriptor = imagine_server::h3_protocol::continuation::Descriptor {
                    schema_version: 1,
                    shape: p.shape,
                    sampled_frames: p.shape.frames,
                    visible_end_frame: p.shape.frames,
                    selected_end_frame: p.shape.frames,
                    handoff_upscale: p.sampling.as_ref().and_then(
                        imagine_server::h3_protocol::continuation::HandoffUpscale::from_sampling,
                    ),
                };
                let bytes = vec![0.125f32; descriptor.bytes() / 4]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect();
                imagine_server::h3_protocol::continuation::Artifact { descriptor, bytes }
            });
            if let Some(context) = &p.continuation {
                anyhow::ensure!(
                    context.valid_sampling(p.shape, p.sampling.as_ref())
                        && context.descriptor.valid_bytes(&p.continuation_bytes)
                );
                anyhow::ensure!(
                    p.continuation_bytes
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()) == 0.125)
                );
            }
            Ok(imagine_server::short_av::Av {
                continuation,
                shape: p.shape,
                frames: vec![vec![70; p.shape.frame_bytes()]; p.shape.frames as usize],
                pcm: vec![0; p.shape.audio_samples() as usize * 8],
            })
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            self.0.stops.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}
#[path = "support/video_boundary.rs"]
mod boundary_tests;
#[path = "support/video_endpoint.rs"]
mod endpoint_tests;
#[path = "support/video_motion.rs"]
mod motion_tests;
#[path = "support/video_runs.rs"]
mod run_tests;

#[tokio::test]
async fn i2v_decodes_the_frozen_staged_image_even_when_its_shared_cursor_is_at_eof() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "i2v").await;
    let (project, mut admission) = scene(&h).await;
    let mut png = Cursor::new(vec![]);
    image::RgbImage::from_pixel(16, 16, image::Rgb([50, 80, 120]))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/video/assets")
                .header("X-Imagine-Session", &h.store.session.id)
                .header("Content-Type", "image/png")
                .body(Body::from(png.into_inner()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 201);
    let asset: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1_000_000).await.unwrap()).unwrap();
    let (code, attached) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/assets"),
        json!({"schema_version":1,"expected_project_revision":1,"asset_id":asset["id"]}),
        None,
    )
    .await;
    assert_eq!(code, 200, "{attached}");
    let asset_id = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap();
    let mut model = attached["project"]["default_model"].clone();
    model["variant"] = json!("i2v");
    model["profile"] = json!(imagine_server::video::job::I2V_PROFILE);
    let (code, edited) = call(&h, "POST", &format!("/v1/video/projects/{project}/edits"), json!({"schema_version":1,"expected_project_revision":2,"commands":[
        {"operation":"set_scene_model","scene_id":"s1","model":{"override":model}},
        {"operation":"set_model_settings","scene_id":"s1","settings":{"model":model,"schema_version":1,"values":{"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}}},
        {"operation":"set_bindings","scene_id":"s1","bindings":[{"id":"own-first","asset":asset_id,"role":"first_frame","enabled":true}]}
    ]}), None).await;
    assert_eq!(code, 200, "{edited}");
    admission["expected_project_revision"] = json!(3);
    admission["seed"] = json!(43);
    let job = submit(&h, admission, "i2v-staged-eof").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    let result = terminal(&h, &job).await;
    assert_eq!(result["status"], "succeeded", "{result}");
    assert_eq!(
        snapshot(&h, &project).await["takes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    stop(h).await;
}
#[tokio::test]
async fn queued_reference_picture_and_voice_survive_later_project_edits() {
    exercise_frozen_reference_media(false).await;
}
#[tokio::test]
async fn queued_timeline_reference_keeps_its_source_range_after_clip_removal() {
    exercise_frozen_reference_media(true).await;
}
async fn exercise_frozen_reference_media(timeline_audio: bool) {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "frozen-reference").await;
    let blocker = submit(&h, image("own queue blocker"), "reference-blocker").await;
    wait_calls(&h, 1).await;
    let (project, mut admission) = scene(&h).await;
    let mut png = Cursor::new(vec![]);
    image::RgbImage::from_pixel(16, 16, image::Rgb([50, 80, 120]))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let pcm: Vec<u8> = [0.125f32, -0.25]
        .repeat(441)
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut wav = b"RIFF".to_vec();
    wav.extend((36 + pcm.len() as u32).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16u32.to_le_bytes());
    wav.extend(3u16.to_le_bytes());
    wav.extend(2u16.to_le_bytes());
    wav.extend(44100u32.to_le_bytes());
    wav.extend(352800u32.to_le_bytes());
    wav.extend(8u16.to_le_bytes());
    wav.extend(32u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend((pcm.len() as u32).to_le_bytes());
    wav.extend(&pcm);
    let mut assets = vec![];
    let mut revision = 1;
    for (mime, bytes) in [("image/png", png.into_inner()), ("audio/wav", wav)] {
        let response = h
            .router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/video/assets")
                    .header("X-Imagine-Session", &h.store.session.id)
                    .header("Content-Type", mime)
                    .body(Body::from(bytes))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 201);
        let uploaded: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1_000_000).await.unwrap())
                .unwrap();
        let before = snapshot(&h, &project).await["project"]["assets"]
            .as_object()
            .unwrap()
            .clone();
        let (code, attached) = call(&h, "POST", &format!("/v1/video/projects/{project}/assets"),
            json!({"schema_version":1,"expected_project_revision":revision,"asset_id":uploaded["id"]}), None).await;
        assert_eq!(code, 200, "{attached}");
        assets.push(
            attached["project"]["assets"]
                .as_object()
                .unwrap()
                .keys()
                .find(|id| !before.contains_key(*id))
                .unwrap()
                .clone(),
        );
        revision += 1;
    }
    let creative = json!({"references":[{"id":"keeper_id","name":"Own keeper","alias":"Keeper A!","kind":"character","description":"own adult keeper","enabled":true,"strength":1.,"retention":"","images":[{"id":"own-image","asset":assets[0],"enabled":true,"preview":false}],"voice":{"language":"English","description":"own voice","assets":[assets[1]]},"origin":null}],"sections":{}});
    let mut commands = json!([{"operation":"add_section","scene_id":"s1","section":{"id":"own-dialogue","start":{"num":0,"den":1},"duration":{"num":1,"den":1},"prompt":"@keeper_a says [English] \"Hello!\""}},{"operation":"set_creative","creative":creative}]);
    if timeline_audio {
        let mut clip = imagine_server::video::contract::AudioClip::new(
            imagine_server::video::contract::AudioClipId("own-timeline-reference".into()),
            imagine_server::video::contract::Time { num: 0, den: 1 },
            imagine_server::video::contract::Time { num: 1, den: 1 },
            &Default::default(),
        );
        clip.asset = Some(imagine_server::video::contract::AssetId(assets[1].clone()));
        clip.source_in = imagine_server::video::contract::Time {
            num: 25,
            den: 10000,
        };
        clip.source_out = Some(imagine_server::video::contract::Time {
            num: 75,
            den: 10000,
        });
        clip.usage.generation_reference = true;
        clip.usage.final_mix = false;
        commands.as_array_mut().unwrap().push(
            json!({"operation":"set_audio_tracks","tracks":[{"id":"own-track","clips":[clip]}]}),
        );
    }
    let (code, edited) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        json!({"schema_version":1,"expected_project_revision":revision,"commands":commands}),
        None,
    )
    .await;
    assert_eq!(code, 200, "{edited}");
    revision += 1;
    admission["expected_project_revision"] = json!(revision);
    let job = submit(&h, admission, "frozen-reference-job").await;
    let (code, edited) = call(&h, "POST", &format!("/v1/video/projects/{project}/edits"),
        json!({"schema_version":1,"expected_project_revision":revision,"commands":[{"operation":"set_creative","creative":{"references":[],"sections":{}}},{"operation":"set_audio_tracks","tracks":[]}]}), None).await;
    assert_eq!(code, 200, "{edited}");
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &blocker).await["status"], "succeeded");
    wait_calls(&h, 2).await;
    {
        let captured = h.probe.references.lock().unwrap();
        let input = &captured[0];
        assert_eq!(input.images.len(), 1);
        assert_eq!((input.images[0].width, input.images[0].height), (16, 16));
        assert!(
            input.images[0]
                .rgb
                .chunks_exact(3)
                .all(|pixel| pixel == [50, 80, 120])
        );
        assert_eq!(input.audios.len(), if timeline_audio { 2 } else { 1 });
        assert_eq!(
            (input.audios[0].sample_rate, input.audios[0].samples),
            (44100, 441)
        );
        assert_eq!(input.audios[0].pcm, pcm);
        assert_eq!(input.manifest["images"].as_array().unwrap().len(), 1);
        assert_eq!(
            input.manifest["audios"].as_array().unwrap().len(),
            if timeline_audio { 2 } else { 1 }
        );
        if timeline_audio {
            assert_eq!(input.audios[1].samples, 221);
            assert_eq!(input.audios[1].pcm, pcm[110 * 8..331 * 8]);
            assert_eq!(
                input.manifest["audios"][1]["role"],
                "standalone_timeline_audio_reference"
            );
        }
    }
    h.probe.release.notify_one();
    let result = terminal(&h, &job).await;
    assert_eq!(result["status"], "succeeded", "{result}");
    stop(h).await;
}

struct Harness {
    router: Router,
    store: Store,
    control: Control,
    task: JoinHandle<anyhow::Result<()>>,
    broker: JoinHandle<()>,
    probe: Arc<Probe>,
}
async fn setup(root: &Path, name: &str) -> Harness {
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        net::UnixListener,
        task::JoinSet,
    };
    let socket = root.join(format!("{name}.sock"));
    let listener = UnixListener::bind(&socket).unwrap();
    let semaphore = Arc::new(Semaphore::new(1));
    let broker = tokio::spawn(async move {
        let mut peers = JoinSet::new();
        loop {
            tokio::select! {
                accepted=listener.accept()=>{let(stream,_)=accepted.unwrap();let semaphore=semaphore.clone();peers.spawn(async move {
                    let (reader,mut writer)=stream.into_split();let mut reader=BufReader::new(reader);let mut line=String::new();
                    if reader.read_line(&mut line).await.unwrap_or(0)==0 {return;}
                    writer.write_all(b"{\"queued\":true}\n").await.unwrap();
                    let _permit=semaphore.acquire_owned().await.unwrap();
                    if writer.write_all(b"{\"granted\":true}\n").await.is_err() {return;}
                    line.clear();let _=reader.read_line(&mut line).await;
                });},
                _=peers.join_next(),if !peers.is_empty()=>{},
            }
        }
    });
    let data = root.join(name);
    std::fs::create_dir_all(&data).unwrap();
    let mut store = Store::open(&data).await.unwrap();
    store.videos.destination = Some(destination::destination(&root.join("output")));
    let probe = Arc::new(Probe::default());
    let (control, task) =
        scheduler::start_coordinated(store.clone(), Box::new(Fake(probe.clone())), Some(socket))
            .await
            .unwrap();
    let (_, engine) = watch::channel(EngineState {
        backend: "rust".into(),
        h3_available: true,
        ..Default::default()
    });
    let router = api::router(App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(data.join("enhancement.json"))
            .unwrap(),
        library: Default::default(),
        profiles: Arc::new(
            imagine_server::profiles::Catalog::load(Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/profiles"
            )))
            .unwrap(),
        ),
        store: store.clone(),
        control: control.clone(),
        engine,
        uploads: Arc::new(Semaphore::new(2)),
    });
    Harness {
        router,
        store,
        control,
        task,
        broker,
        probe,
    }
}
async fn stop(h: Harness) {
    h.control.call(Action::Shutdown).await.unwrap();
    h.task.await.unwrap().unwrap();
    h.broker.abort();
    let _ = h.broker.await;
    h.store.close().await.unwrap();
}
async fn call(
    h: &Harness,
    method: &str,
    path: &str,
    value: Value,
    key: Option<&str>,
) -> (u16, Value) {
    let mut r = Request::builder()
        .method(method)
        .uri(path)
        .header("Content-Type", "application/json")
        .header("X-Imagine-Session", &h.store.session.id);
    if let Some(key) = key {
        r = r.header("Idempotency-Key", key);
    }
    let response = h
        .router
        .clone()
        .oneshot(
            r.body(if value.is_null() {
                Body::empty()
            } else {
                Body::from(value.to_string())
            })
            .unwrap(),
        )
        .await
        .unwrap();
    let code = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    (code, serde_json::from_slice(&bytes).unwrap())
}
async fn wait_calls(h: &Harness, count: usize) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while h.probe.calls.lock().unwrap().len() < count {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
async fn terminal(h: &Harness, id: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let v = call(h, "GET", &format!("/v1/jobs/{id}"), Value::Null, None)
                .await
                .1;
            if matches!(
                v["status"].as_str(),
                Some("succeeded" | "failed" | "cancelled" | "interrupted")
            ) {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
fn image(prompt: &str) -> Value {
    json!({"model":imagine_server::contract::MODEL,"operation":"text_to_image","prompt":prompt,"seed":42,"output":{"format":"png","count":1}})
}
async fn scene(h: &Harness) -> (String, Value) {
    let model = json!({"adapter":"h3","model":"minimax-h3","variant":"reference","profile":imagine_server::video::job::PROFILE,"profile_revision":"2"});
    let (s,p)=call(h,"POST","/v1/video/projects",json!({"schema_version":1,"default_model":model,"delivery":{"width":32,"height":32,"fps":{"num":24,"den":1}}}),None).await;
    assert_eq!(s, 201, "{p}");
    let id = p["project"]["id"].as_str().unwrap().to_owned();
    let s = json!({"id":"s1","start":{"num":0,"den":1},"duration":{"num":1,"den":1},"description":"own original scene","model":"project_default","settings":[{"model":model,"schema_version":1,"values":{"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}}],"sections":[],"bindings":[],"continuity":null,"accepted_take":null});
    let (code,v)=call(h,"POST",&format!("/v1/video/projects/{id}/edits"),json!({"schema_version":1,"expected_project_revision":0,"commands":[{"operation":"add_scene","scene":s}]}),None).await;
    assert_eq!(code, 200, "{v}");
    let admission = json!({"kind":"video_scene","schema_version":1,"project_id":id,"expected_project_revision":1,"scene_id":"s1","seed":42});
    (id, admission)
}
async fn submit(h: &Harness, value: Value, key: &str) -> String {
    let (s, v) = call(h, "POST", "/v1/jobs", value, Some(key)).await;
    assert_eq!(s, 202, "{v}");
    v["id"].as_str().unwrap().into()
}
async fn snapshot(h: &Harness, id: &str) -> Value {
    call(
        h,
        "GET",
        &format!("/v1/video/projects/{id}"),
        Value::Null,
        None,
    )
    .await
    .1
}
#[tokio::test]
async fn image_scene_image_fifo_frozen_inputs_idempotency_and_private_events() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state").await;
    let first = submit(&h, image("own first image"), "image1").await;
    wait_calls(&h, 1).await;
    let (project, request) = scene(&h).await;
    let job = submit(&h, request.clone(), "scene").await;
    let last = submit(&h, image("own last image"), "image2").await;
    let(s,_)=call(&h,"POST",&format!("/v1/video/projects/{project}/edits"),json!({"schema_version":1,"expected_project_revision":1,"commands":[{"operation":"set_scene_description","scene_id":"s1","description":"own edited later"}]}),None).await;
    assert_eq!(s, 200);
    let (s, replay) = call(&h, "POST", "/v1/jobs", request.clone(), Some("scene")).await;
    assert_eq!(s, 202);
    assert_eq!(replay["id"], job);
    assert_eq!(replay["reused"], true);
    let mut changed = request;
    changed["seed"] = json!(43);
    assert_eq!(
        call(&h, "POST", "/v1/jobs", changed, Some("scene")).await.0,
        409
    );
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &first).await["status"], "succeeded");
    wait_calls(&h, 2).await;
    let frozen_prompt = h.probe.calls.lock().unwrap()[1].clone();
    assert!(frozen_prompt.contains("\ndetailed_description:\nown original scene\n"));
    assert!(!frozen_prompt.contains("own edited later"));
    h.probe.release.notify_one();
    let done = terminal(&h, &job).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    wait_calls(&h, 3).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &last).await["status"], "succeeded");
    assert_eq!(
        *h.probe.calls.lock().unwrap(),
        [
            "own first image".to_owned(),
            frozen_prompt,
            "own last image".to_owned()
        ]
    );
    let project = snapshot(&h, &project).await;
    assert_eq!(project["takes"].as_array().unwrap().len(), 1);
    assert_eq!(project["takes"][0]["status"], "outdated");
    assert!(project["project"]["scenes"][0]["accepted_take"].is_null());
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/jobs/{job}/events"))
                .header("X-Imagine-Session", &h.store.session.id)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let text = String::from_utf8(
        to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(!text.contains("own original"));
    assert!(!text.contains("own edited"));
    assert!(text.contains("video_scene"));
    stop(h).await;
}
#[tokio::test]
async fn queued_and_active_cancel_publish_no_take_and_failed_job_requires_explicit_retry() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state").await;
    let (project, request) = scene(&h).await;
    let first = submit(&h, request.clone(), "active").await;
    wait_calls(&h, 1).await;
    let queued = submit(&h, request.clone(), "queued").await;
    for id in [&queued, &first] {
        assert_eq!(
            call(
                &h,
                "POST",
                &format!("/v1/jobs/{id}/cancel"),
                Value::Null,
                None
            )
            .await
            .0,
            202
        );
        assert_eq!(terminal(&h, id).await["status"], "cancelled");
    }
    assert!(
        snapshot(&h, &project).await["takes"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(h.probe.calls.lock().unwrap().len(), 1);
    assert!(h.probe.stops.load(Ordering::SeqCst) > 0);
    h.probe.fail.store(true, Ordering::SeqCst);
    let failed = submit(&h, request.clone(), "failure").await;
    wait_calls(&h, 2).await;
    h.probe.release.notify_one();
    let ended = terminal(&h, &failed).await;
    assert_eq!(ended["status"], "failed");
    // Bounded worker diagnostics reach the job error; nothing else does.
    assert_eq!(
        ended["error"],
        json!({"code":"h3_generation_failed","stage":"preparing_motion_context","kind":"shape_mismatch","sizes":[7,8]})
    );
    assert_eq!(submit(&h, request.clone(), "failure").await, failed);
    assert_eq!(h.probe.calls.lock().unwrap().len(), 2);
    let retry = submit(&h, request, "explicit-retry").await;
    wait_calls(&h, 3).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &retry).await["status"], "succeeded");
    assert_eq!(
        snapshot(&h, &project).await["takes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    stop(h).await;
}
#[tokio::test]
async fn successful_take_survives_autosave_failure_retry_and_reopen_after_output_deletion() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state1").await;
    let (project, request) = scene(&h).await;
    let folder = root.path().join("project");
    std::fs::create_dir(&folder).unwrap();
    let (_, loc) = call(
        &h,
        "POST",
        "/v1/video/locations",
        json!({"schema_version":1,"path":folder}),
        None,
    )
    .await;
    assert_eq!(call(&h,"POST",&format!("/v1/video/projects/{project}/save"),json!({"schema_version":1,"expected_project_revision":1,"location_id":loc["location_id"]}),None).await.0,200);
    let job = submit(&h, request, "take").await;
    wait_calls(&h, 1).await;
    let moved = root.path().join("moved");
    std::fs::rename(&folder, &moved).unwrap();
    std::fs::create_dir(&folder).unwrap();
    h.probe.release.notify_one();
    let done = terminal(&h, &job).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    let draft = snapshot(&h, &project).await;
    assert_eq!(draft["persistence"]["state"], "not_saved");
    assert_eq!(draft["takes"].as_array().unwrap().len(), 1);
    std::fs::remove_dir(&folder).unwrap();
    std::fs::rename(&moved, &folder).unwrap();
    let (s, saved) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/save"),
        json!({"schema_version":1,"expected_project_revision":2}),
        None,
    )
    .await;
    assert_eq!(s, 200, "{saved}");
    let old_asset = done["output"]["assets"][0]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let output = done["output"]["saved_path"].as_str().unwrap().to_owned();
    std::fs::remove_file(&output).unwrap();
    let (code, media) = call(
        &h,
        "GET",
        &format!("/v1/video/assets/{old_asset}"),
        Value::Null,
        None,
    )
    .await;
    assert_eq!(code, 200, "{media}");
    assert_eq!(media["media"]["kind"], "video");
    stop(h).await;
    let h = setup(root.path(), "state2").await;
    assert_eq!(
        call(&h, "GET", &format!("/v1/jobs/{job}"), Value::Null, None)
            .await
            .0,
        404
    );
    let (_, loc) = call(
        &h,
        "POST",
        "/v1/video/locations",
        json!({"schema_version":1,"path":folder}),
        None,
    )
    .await;
    let (s, opened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        json!({"schema_version":1,"location_id":loc["location_id"]}),
        None,
    )
    .await;
    assert_eq!(s, 200, "{opened}");
    let take = opened["takes"][0]["id"].as_str().unwrap();
    let asset = opened["runtime_takes"][take].as_str().unwrap();
    assert_ne!(asset, old_asset);
    let (s, info) = call(
        &h,
        "GET",
        &format!("/v1/video/assets/{asset}"),
        Value::Null,
        None,
    )
    .await;
    assert_eq!(s, 200, "{info}");
    assert_eq!(info["media"]["kind"], "video");
    assert_eq!(info["media"]["audio"], true);
    stop(h).await;
}
#[tokio::test]
async fn failed_output_transaction_never_commits_a_take_and_cancel_result_races_are_consistent() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state").await;
    let (project, mut request) = scene(&h).await;
    sqlx::raw_sql("CREATE TRIGGER own_fail_success BEFORE UPDATE ON jobs WHEN NEW.status='succeeded' BEGIN SELECT RAISE(ABORT,'own synthetic SQL failure'); END;").execute(&h.store.db).await.unwrap();
    let job = submit(&h, request.clone(), "fail-output-commit").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    let failed = terminal(&h, &job).await;
    assert_eq!(failed["status"], "failed");
    assert!(failed["output"].is_null());
    assert!(
        snapshot(&h, &project).await["takes"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM assets")
            .fetch_one(&h.store.db)
            .await
            .unwrap(),
        0
    );
    sqlx::raw_sql("DROP TRIGGER own_fail_success;")
        .execute(&h.store.db)
        .await
        .unwrap();
    // Consume the first worker result notification; subsequent iterations cancel
    // as raw generation completes and encoding/publication can become ready.
    h.probe.produced.notified().await;
    let mut completed = 0;
    for n in 0..4 {
        request["expected_project_revision"] =
            snapshot(&h, &project).await["project"]["project_revision"].clone();
        let job = submit(&h, request.clone(), &format!("race-{n}")).await;
        wait_calls(&h, n + 2).await;
        h.probe.release.notify_one();
        h.probe.produced.notified().await;
        let (code, _) = call(
            &h,
            "POST",
            &format!("/v1/jobs/{job}/cancel"),
            Value::Null,
            None,
        )
        .await;
        let result = terminal(&h, &job).await;
        if code == 202 {
            assert_eq!(result["status"], "cancelled");
            assert!(result["output"].is_null());
        } else {
            assert_eq!(code, 409);
            assert_eq!(result["status"], "succeeded");
            completed += 1;
        }
        assert_eq!(
            snapshot(&h, &project).await["takes"]
                .as_array()
                .unwrap()
                .len(),
            completed
        );
    }
    stop(h).await;
}
#[tokio::test]
async fn shutdown_and_coordinator_loss_release_owned_work_without_publishing_candidates() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state1").await;
    let (project, request) = scene(&h).await;
    let active = submit(&h, request.clone(), "active").await;
    wait_calls(&h, 1).await;
    let queued = submit(&h, request, "queued").await;
    let image = submit(&h, image("own queued image"), "image").await;
    h.control.call(Action::Shutdown).await.unwrap();
    for id in [&active, &queued, &image] {
        let v = h.store.job(id, "local").await.unwrap().unwrap();
        assert_eq!(v["status"], "interrupted");
        assert!(v["output"].is_null());
    }
    assert!(
        snapshot(&h, &project).await["takes"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    h.task.await.unwrap().unwrap();
    h.broker.abort();
    let _ = h.broker.await;
    h.store.close().await.unwrap();
    let h = setup(root.path(), "state2").await;
    let (project, request) = scene(&h).await;
    let active = submit(&h, request.clone(), "active").await;
    wait_calls(&h, 1).await;
    let queued = submit(&h, request, "queued").await;
    h.broker.abort();
    for id in [&active, &queued] {
        let v = terminal(&h, id).await;
        assert_eq!(v["status"], "failed");
        assert_eq!(v["error"]["code"], "coordinator_unavailable");
    }
    assert!(
        snapshot(&h, &project).await["takes"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(h.probe.stops.load(Ordering::SeqCst) > 0);
    stop(h).await;
}
#[tokio::test]
async fn full_u64_seed_and_inactive_trained_upscaler_metadata_reach_normal_admission() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state").await;
    let (project, mut request) = scene(&h).await;
    request["seed"] = json!(u64::MAX);
    let first = submit(&h, request.clone(), "full-u64").await;
    wait_calls(&h, 1).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &first).await["status"], "succeeded");
    let before = snapshot(&h, &project).await;
    assert_eq!(before["takes"][0]["provenance"]["seed"], json!(u64::MAX));
    request["seed"] = json!(42);
    let model = before["project"]["default_model"].clone();
    let sampling = json!({"steps":20,"sampler":"res_multistep","scheduler":"simple"});
    let(s,v)=call(&h,"POST",&format!("/v1/video/projects/{project}/edits"),json!({"schema_version":1,"expected_project_revision":before["project"]["project_revision"],"commands":[{"operation":"set_model_settings","scene_id":"s1","settings":{"model":model,"schema_version":1,"values":{"stage_one":sampling,"stage_two":{"sampling":sampling,"model":model,"handoff":"continue_latent","trained_latent_upscale_model":"explicit-unused-test-upscaler"}}}}]}),None).await;
    assert_eq!(s, 200, "{v}");
    request["expected_project_revision"] = v["project"]["project_revision"].clone();
    let second = submit(&h, request, "inactive-upscaler").await;
    wait_calls(&h, 2).await;
    h.probe.release.notify_one();
    assert_eq!(terminal(&h, &second).await["status"], "succeeded");
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM jobs")
            .fetch_one(&h.store.db)
            .await
            .unwrap(),
        2
    );
    stop(h).await;
}

#[tokio::test]
async fn manual_take_review_conflicts_invalidation_and_reopen_keep_media_and_provenance() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "review1").await;
    let (project, mut request) = scene(&h).await;
    for index in 0..2 {
        request["expected_project_revision"] = json!(1 + index);
        let job = submit(&h, request.clone(), &format!("review-take-{index}")).await;
        wait_calls(&h, index as usize + 1).await;
        h.probe.release.notify_one();
        assert_eq!(terminal(&h, &job).await["status"], "succeeded");
    }
    let initial = snapshot(&h, &project).await;
    let first = initial["takes"][0]["id"].clone();
    let second = initial["takes"][1]["id"].clone();
    let review = format!("/v1/video/projects/{project}/review");
    let body = |revision, take: &Value, action| json!({"schema_version":1,"expected_project_revision":revision,"scene_id":"s1","take_id":take,"action":action});
    let (code, missing) = call(
        &h,
        "POST",
        &review,
        body(3, &json!("missing-own-take"), "accept"),
        None,
    )
    .await;
    assert_eq!(code, 404, "{missing}");
    assert_eq!(missing["snapshot"]["project"]["project_revision"], 3);
    let (code, accepted) = call(&h, "POST", &review, body(3, &first, "accept"), None).await;
    assert_eq!(code, 200, "{accepted}");
    assert_eq!(
        accepted["project"]["scenes"][0]["accepted_take"]["id"],
        first
    );
    assert_eq!(
        accepted["take_states"][first.as_str().unwrap()]["accepted"],
        true
    );
    assert_eq!(
        accepted["take_states"][first.as_str().unwrap()]["outdated"],
        false
    );
    let (code, stale) = call(&h, "POST", &review, body(3, &second, "accept"), None).await;
    assert_eq!(code, 409);
    assert_eq!(stale["snapshot"]["project"]["project_revision"], 4);
    let (code, rejected) = call(&h, "POST", &review, body(4, &second, "reject"), None).await;
    assert_eq!(code, 200, "{rejected}");
    assert_eq!(
        rejected["project"]["scenes"][0]["accepted_take"]["id"],
        first
    );
    let (code, edited) = call(&h, "POST", &format!("/v1/video/projects/{project}/edits"), json!({"schema_version":1,"expected_project_revision":5,"commands":[{"operation":"set_scene_description","scene_id":"s1","description":"Changed synthetic scene"}]}), None).await;
    assert_eq!(code, 200, "{edited}");
    assert_eq!(edited["project"]["scenes"][0]["accepted_take"]["id"], first);
    assert_eq!(edited["takes"][0]["status"], "accepted");
    assert_eq!(edited["takes"][1]["status"], "rejected");
    let (code, outdated) = call(&h, "POST", &review, body(6, &first, "outdated"), None).await;
    assert_eq!(code, 200, "{outdated}");
    assert!(outdated["project"]["scenes"][0]["accepted_take"].is_null());
    let (code, old) = call(&h, "POST", &review, body(7, &first, "accept"), None).await;
    assert_eq!(code, 200, "{old}");
    assert_eq!(
        old["take_states"][first.as_str().unwrap()],
        json!({"accepted":true,"outdated":false})
    );
    assert_eq!(
        old["takes"][0]["provenance"],
        initial["takes"][0]["provenance"]
    );
    let folder = root.path().join("review-project");
    std::fs::create_dir(&folder).unwrap();
    let (_, location) = call(
        &h,
        "POST",
        "/v1/video/locations",
        json!({"schema_version":1,"path":folder}),
        None,
    )
    .await;
    let (code, saved) = call(&h, "POST", &format!("/v1/video/projects/{project}/save"), json!({"schema_version":1,"expected_project_revision":8,"location_id":location["location_id"]}), None).await;
    assert_eq!(code, 200, "{saved}");
    stop(h).await;
    let h = setup(root.path(), "review2").await;
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
    // Every snapshot response parses into the shared client view.
    for snapshot in [&accepted, &rejected, &edited, &old, &saved, &reopened] {
        let (project, view) = imagine_video_contract::snapshot::parse(snapshot)
            .unwrap_or_else(|| panic!("client view rejects {snapshot}"));
        assert_eq!(view.takes.len(), 2);
        assert_eq!(view.take_states.len(), 2);
        assert!(project.scenes.iter().any(|s| s.id.0 == "s1"));
    }
    let view = imagine_video_contract::snapshot::parse(&saved).unwrap().1;
    assert_eq!(
        view.persistence.unwrap().state,
        imagine_video_contract::snapshot::SaveState::Saved
    );
    assert_eq!(reopened["takes"], saved["takes"]);
    assert_eq!(reopened["take_states"], saved["take_states"]);
    assert_eq!(reopened["runtime_takes"].as_object().unwrap().len(), 2);
    assert_ne!(reopened["runtime_takes"], saved["runtime_takes"]);
    // Project Take cleanup previews, refuses a stale revision, then clears.
    let cleanup = format!(
        "/v1/video/projects/{}/takes/cleanup",
        reopened["project"]["id"].as_str().unwrap()
    );
    let revision = reopened["project"]["project_revision"].as_u64().unwrap();
    let body = |revision: u64, apply: bool| json!({"schema_version":1,"expected_project_revision":revision,"apply":apply});
    let (code, preview) = call(&h, "POST", &cleanup, body(revision, false), None).await;
    assert_eq!(code, 200, "{preview}");
    assert_eq!(preview["preview"]["takes"], 2);
    assert!(
        preview["preview"]["owned_files"].as_u64().unwrap() > 0,
        "{preview}"
    );
    let (code, _) = call(&h, "POST", &cleanup, body(revision - 1, true), None).await;
    assert_eq!(code, 409);
    let (code, cleared) = call(&h, "POST", &cleanup, body(revision, true), None).await;
    assert_eq!(code, 200, "{cleared}");
    assert_eq!(cleared["takes"], json!([]));
    assert!(cleared["project"]["scenes"][0]["accepted_take"].is_null());
    assert!(cleared["runtime_takes"].as_object().unwrap().is_empty());
    assert!(
        cleared["cleanup_result"]["cleanup_warning"].is_null(),
        "{cleared}"
    );
    stop(h).await;
}
