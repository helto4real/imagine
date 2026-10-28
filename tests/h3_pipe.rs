//! Synthetic process/pipe lifecycle only; no models, CUDA, service or playback.
use imagine_server::{
    comfy_engine::Comfy,
    config::Config,
    engine::{Backend, EngineState},
    generation::ImagePass,
    h3_protocol as h3,
    storage::hash,
};
use serde_json::json;
use std::{collections::BTreeMap, os::unix::fs::PermissionsExt, path::Path, time::Duration};
use tokio::sync::watch;
fn input(prompt: &str) -> h3::Request {
    h3::Request {
        preview: None,
        references: Default::default(),
        options: Default::default(),
        capture_continuation: false,
        continuation: None,
        continuation_bytes: vec![],
        sampling: None,
        protocol: h3::VERSION,
        kind: h3::RequestKind::GenerateVideo,
        id: "fixture-42".into(),
        profile: h3::Profile::ShortAvNoReferenceV1,
        prompt: prompt.into(),
        shape: Default::default(),
        image: None,
        image_rgb: Vec::new(),
        seed: 42,
    }
}
fn fake(mismatch: bool) -> (tempfile::TempDir, Comfy, watch::Receiver<EngineState>) {
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("image-worker");
    let video = dir.path().join("h3-worker");
    let dep = dir.path().join("dependency");
    std::fs::write(&dep, b"synthetic dependency, never loaded").unwrap();
    let prefix = r#"#!/usr/bin/python3
import sys,json,struct,time,os
r=sys.stdin.buffer;w=sys.stdout.buffer
def read():
 n=struct.unpack('>I',r.read(4))[0];return r.read(n)
def emit(e,b=b''):
 h=json.dumps(e).encode();w.write(struct.pack('>I',len(h))+h+struct.pack('>I',len(b))+b);w.flush()
"#;
    std::fs::write(
        &image,
        format!(
            "{prefix}{}",
            r#"
emit(dict(type='ready',protocol=1,attention='checkpoint_preferences'))
while True:
 try:q=json.loads(read());body=read()
 except Exception:break
 emit(dict(type='result',id=q['id']),bytes([0,0,0,255])*q['width']*q['height'])
"#
        ),
    )
    .unwrap();
    let bad = if mismatch {
        "d['worker_sha256']='0'*64"
    } else {
        "pass"
    };
    std::fs::write(&video,format!("{prefix}\nd=json.loads(os.environ['IMAGINE_H3_DEPLOYMENT'])\n{bad}\n{}",r#"
emit(dict(type='ready',deployment=d))
q=json.loads(read());body=read()
if q['profile']=='h3-fl2va-i2va-v1':
 import hashlib
 im=q['image'];assert len(body)==im['width']*im['height']*3
 assert hashlib.sha256(body).hexdigest()==im['sha256']
else: assert not body
assert q['kind']=='generate_video' and q['seed']==42
if q['prompt']=='hang':time.sleep(60)
if q['prompt']=='failure':
 emit(dict(type='error',id=q['id'],code='generation_failed'));sys.exit(0)
if q['prompt']=='diagnosed':
 emit(dict(type='error',id=q['id'],code='generation_failed',failure=dict(stage='sampling',kind='shape_mismatch',exception='RuntimeError',sizes=[7,8,2])));sys.exit(0)
if q['prompt']=='overlong':
 emit(dict(type='error',id=q['id'],code='generation_failed',failure=dict(stage='sampling',exception='Not a class name')));sys.exit(0)
if q['prompt']=='oversize':
 h=json.dumps(dict(type='video_frame',id=q['id'],index=0)).encode()
 w.write(struct.pack('>I',len(h))+h+struct.pack('>I',4194305));w.flush();time.sleep(60)
if q['prompt']=='unknown':
 emit(dict(type='error',id=q['id'],code='generation_failed',message='PRIVATE SYNTHETIC ERROR'));sys.exit(0)
if q['prompt'].startswith('cache-'):
 emit(dict(type='reference_cache',id=q['id'],command=dict(operation='lookup',key='a'*64)))
 reply=json.loads(read());cached=read()
 if q['prompt']=='cache-hit':
  assert reply['state']=='chunk' and cached==bytes([1,2,3,4])
 elif q['prompt']=='cache-off':
  assert reply==dict(state='miss',ticket=None) and not cached
 else:
  assert reply['state']=='miss' and reply['ticket'] is not None and not cached
  emit(dict(type='reference_cache',id=q['id'],command=dict(operation='store',ticket=reply['ticket'],descriptor=dict(shape=[2],dtype='f16'),offset=0)),bytes([1,2,3,4]))
  reply=json.loads(read());assert not read()
  assert reply==dict(state='stored',accepted=True,complete=True)
for phase in ['loading_text','encoding_text','loading_dit','sampling','loading_video_vae','decoding_video','loading_audio_vae','decoding_audio']:
 emit(dict(type='phase',id=q['id'],phase=phase))
for i in range(1,21):emit(dict(type='progress',id=q['id'],completed=i,total=20,elapsed_seconds=i/10))
if q['prompt']=='preview':
 assert q['preview']['enabled']
 emit(dict(type='sampling_preview',id=q['id'],metadata=dict(sequence=1,step=20,width=2,height=1,frames=2,frame_duration_ms=812,source_duration_seconds=39/24)),bytes([7])*12)
 emit(dict(type='preview_warning',id=q['id'],code='fast_preview_failed'))
 time.sleep(.3)
shape=q['shape']; count=shape['frames']; samples=(count*4000+1)//3
for i in range(count):emit(dict(type='video_frame',id=q['id'],index=i),bytes([i%256])*shape['width']*shape['height']*3)
if q['prompt']=='partial':sys.exit(0)
for i in range(0,samples,1000):emit(dict(type='audio_chunk',id=q['id'],start=i,samples=min(1000,samples-i)),bytes(min(1000,samples-i)*8))
emit(dict(type='complete',id=q['id'],frames=count,audio_samples=samples))
time.sleep(60)
"#)).unwrap();
    for path in [&image, &video] {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let files = BTreeMap::from([
        (image.clone(), hash(&std::fs::read(&image).unwrap())),
        (dep.clone(), hash(&std::fs::read(&dep).unwrap())),
    ]);
    let h3files = BTreeMap::from([
        (video.clone(), hash(&std::fs::read(&video).unwrap())),
        (dep.clone(), hash(&std::fs::read(&dep).unwrap())),
    ]);
    let deps = BTreeMap::from([(dep.clone(), hash(&std::fs::read(&dep).unwrap()))]);
    let environment = BTreeMap::from([
        ("IMAGINE_KITCHEN_LIBRARY", dep.clone()),
        ("IMAGINE_AIMDO_LIBRARY", dep.clone()),
        ("IMAGINE_PILLOW_LIBRARY", dep.clone()),
        ("IMAGINE_TRITON_COMPILER_PROFILE", dep.clone()),
    ]);
    let h3environment = environment.clone();
    let identity = h3::Deployment {
        protocol: h3::VERSION,
        profile: h3::Profile::ShortAvNoReferenceV1,
        core_revision: h3::CORE_REVISION.into(),
        model_lock_sha256: h3::MODEL_LOCK_SHA256.into(),
        worker_sha256: h3files[&video].clone(),
        dependencies_sha256: hash(&serde_json::to_vec(&deps).unwrap()),
    };
    let dependency_map = dir.path().join("dependencies.json");
    std::fs::write(&dependency_map, serde_json::to_vec_pretty(&deps).unwrap()).unwrap();
    let manifest = dir.path().join("runtime.json");
    std::fs::write(&manifest,serde_json::to_vec(&json!({"schema":1,"reference_revision":"446ef4e005fc7134d53ce6869ebd77d2134cd39c",
        "binary":image,"environment":environment,"files":files,"h3":{"profile_checkpoints":h3::profile_checkpoints(),"binary":video,"model_root":dir.path().join("models"),
        "identity":identity,"dependency_map":dependency_map,"files":h3files,"environment":h3environment}})).unwrap()).unwrap();
    let (tx, rx) = watch::channel(EngineState {
        backend: "rust".into(),
        ..Default::default()
    });
    let config = Config {
        project: dir.path().into(),
        data: dir.path().join("session"),
        bind: "127.0.0.1:0".parse().unwrap(),
        model_root: dir.path().join("models"),
        private_storage: None,
    };
    let engine = Comfy::new_rust(config, tx, &manifest).unwrap();
    (dir, engine, rx)
}
fn image() -> ImagePass {
    ImagePass {
        resolved_models: None,
        model: imagine_server::contract::MODEL.into(),
        model_options: Default::default(),
        sampling: Default::default(),
        prompt: "synthetic".into(),
        negative_prompt: String::new(),
        width: 256,
        height: 128,
        seed: 42,
        references: vec![],
        maps: vec![],
    }
}
#[tokio::test]
async fn real_supervised_cache_channel_survives_worker_exit_and_obeys_clear_and_off() {
    let (_dir, mut engine, _) = fake(false);
    let control = engine.reference_cache().unwrap();
    assert_eq!(control.status().unwrap().entries, 0);
    engine.generate_video(&input("cache-miss")).await.unwrap();
    let status = control.status().unwrap();
    assert_eq!((status.entries, status.used_bytes, status.busy), (1, 4, 0));
    engine.generate_video(&input("cache-hit")).await.unwrap();
    assert_eq!(control.status().unwrap().used_bytes, 4);
    control.clear().unwrap();
    assert_eq!(control.status().unwrap().used_bytes, 0);
    engine.generate_video(&input("cache-miss")).await.unwrap();
    control.configure(0, false).unwrap();
    engine.generate_video(&input("cache-off")).await.unwrap();
    assert_eq!(control.status().unwrap().entries, 0);
    engine.stop().await.unwrap();
}
#[tokio::test]
async fn image_video_image_share_one_supervised_slot_and_reap_h3_on_success() {
    let (_dir, mut engine, state) = fake(false);
    assert!(state.borrow().h3_available);
    assert!(state.borrow().pid.is_none());
    engine.generate(&image()).await.unwrap();
    let old = state.borrow().pid.unwrap();
    let av = engine.generate_video(&input("synthetic")).await.unwrap();
    assert!(av.valid());
    assert_eq!(av.frames[38][0], 38);
    assert!(!Path::new(&format!("/proc/{old}")).exists());
    assert!(state.borrow().pid.is_none());
    engine.generate(&image()).await.unwrap();
    assert_ne!(state.borrow().pid.unwrap(), old);
    engine.stop().await.unwrap();
}
#[tokio::test]
async fn mismatch_partial_errors_and_bounds_never_publish_av_and_remain_sanitized() {
    for (mismatch, prompt, code) in [
        (true, "synthetic", "h3_deployment_mismatch"),
        (false, "partial", "h3_protocol_error"),
        (false, "failure", "h3_generation_failed"),
        (false, "unknown", "h3_protocol_error"),
        (false, "oversize", "h3_protocol_error"),
    ] {
        let (_dir, mut engine, state) = fake(mismatch);
        let error = engine
            .generate_video(&input(prompt))
            .await
            .err()
            .expect("must fail");
        assert_eq!(error.to_string(), code);
        assert!(!error.to_string().contains("PRIVATE"));
        let pid = state.borrow().pid.unwrap();
        engine.stop().await.unwrap();
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
    }
}
#[tokio::test]
async fn cancelled_future_leaves_owned_child_for_stop_to_reap_before_release() {
    let (_dir, mut engine, state) = fake(false);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(100),
            engine.generate_video(&input("hang"))
        )
        .await
        .is_err()
    );
    let pid = state.borrow().pid.unwrap();
    engine.stop().await.unwrap();
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
}

#[tokio::test]
async fn nondefault_shape_survives_the_real_pipe_boundary() {
    let (_dir, mut engine, state) = fake(false);
    let mut request = input("synthetic");
    request.profile = h3::Profile::AvNoReferenceV2;
    request.shape = h3::Shape {
        width: 64,
        height: 96,
        frames: 22,
        fps: 24,
    };
    let av = engine.generate_video(&request).await.unwrap();
    assert_eq!(av.shape, request.shape);
    assert!(av.valid());
    assert_eq!(av.frames[21][0], 21);
    assert!(state.borrow().pid.is_none());
}

#[tokio::test]
async fn disposable_preview_crosses_actual_pipe_but_never_changes_final_av() {
    let (_dir, mut engine, state) = fake(false);
    engine.begin_job("public-job");
    let hub = state.borrow().sampling_preview.clone();
    let request = input("preview");
    let observe = async {
        for _ in 0..200 {
            if let Some(clip) = hub.snapshot("public-job", 0) {
                assert_eq!(clip.bytes, [7; 12]);
                assert_eq!(clip.metadata.frames, 2);
                assert!(hub.snapshot("fixture-42", 0).is_none());
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        false
    };
    let (output, observed) = tokio::join!(engine.generate_video(&request), observe);
    let output = output.unwrap();
    assert!(observed);
    assert!(output.valid());
    assert_eq!(output.frames[38][0], 38);
    assert!(output.pcm.iter().all(|v| *v == 0));
    assert!(hub.snapshot("public-job", 0).is_none());
    assert!(state.borrow().pid.is_none());
    engine.mark_idle();
    assert!(hub.status().active_job.is_none());
}

#[tokio::test]
async fn i2v_rgb_is_private_and_profile_switch_reaps_each_worker() {
    let (_dir, mut engine, state) = fake(false);
    let mut r = input("synthetic");
    r.profile = h3::Profile::ImageToAvV1;
    r.shape = h3::Shape {
        width: 64,
        height: 64,
        frames: 5,
        fps: 24,
    };
    r.image_rgb = [20, 30, 40].repeat(4);
    r.image = Some(h3::ImageInput {
        width: 2,
        height: 2,
        sha256: hash(&r.image_rgb),
    });
    let mut invalid = r.clone();
    invalid.image_rgb[0] = 99;
    assert!(engine.generate_video(&invalid).await.is_err());
    assert!(state.borrow().pid.is_none());
    assert!(engine.generate_video(&r).await.unwrap().valid());
    assert!(state.borrow().pid.is_none());
    let mut text = input("synthetic");
    text.profile = h3::Profile::AvNoReferenceV2;
    text.shape = r.shape;
    assert!(engine.generate_video(&text).await.unwrap().valid());
    assert!(state.borrow().pid.is_none());
    engine.generate(&image()).await.unwrap();
    engine.stop().await.unwrap();
    assert!(state.borrow().pid.is_none());
}

#[test]
fn observer_output_is_declared_only_in_the_h3_runtime_contract() {
    let (dir, _engine, _) = fake(false);
    let path = dir.path().join("runtime.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["h3"]["environment"]["IMAGINE_REFERENCE_PROBE_OUTPUT"] =
        json!(dir.path().join("own-observer"));
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(imagine_server::rust_runtime::Runtime::load(&path).is_ok());
    manifest["environment"]["IMAGINE_REFERENCE_PROBE_OUTPUT"] =
        json!("not-an-image-runtime-option");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(imagine_server::rust_runtime::Runtime::load(&path).is_err());
}

#[test]
fn historical_provenance_is_optional_but_required_pillow_remains_checked() {
    let (dir, _engine, state) = fake(false);
    let path = dir.path().join("runtime.json");
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for pillow in [false, true] {
        let mut changed = original.clone();
        if pillow {
            changed["h3"]["environment"]
                .as_object_mut()
                .unwrap()
                .remove("IMAGINE_PILLOW_LIBRARY");
        } else {
            changed["reference_revision"] = "own-compatible-rebuild".into();
            changed["h3"]["identity"]["core_revision"] = "own-compatible-rebuild".into();
            changed["h3"]["identity"]["model_lock_sha256"] = "b".repeat(64).into();
            changed["h3"]["profile_checkpoints"]["h3-fl2va-i2va-v1"] = "0".repeat(64).into();
        }
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert_eq!(
            imagine_server::rust_runtime::Runtime::load(&path).is_err(),
            pillow
        );
        assert!(state.borrow().pid.is_none());
    }
}
#[tokio::test]
async fn worker_failure_diagnostics_cross_the_pipe_bounded_and_invalid_ones_are_dropped() {
    let (_dir, mut engine, _state) = fake(false);
    let error = engine
        .generate_video(&input("diagnosed"))
        .await
        .err()
        .expect("must fail");
    assert_eq!(error.to_string(), "h3_generation_failed");
    let failure = &error
        .downcast_ref::<imagine_server::engine::H3Failure>()
        .expect("diagnostics")
        .0;
    assert_eq!(failure.stage, Some(h3::Stage::Sampling));
    assert_eq!(failure.kind, Some(h3::FailureKind::ShapeMismatch));
    assert_eq!(failure.exception.as_deref(), Some("RuntimeError"));
    assert_eq!(failure.sizes, vec![7, 8, 2]);
    engine.stop().await.unwrap();
    let (_dir, mut engine, _state) = fake(false);
    let error = engine
        .generate_video(&input("overlong"))
        .await
        .err()
        .expect("must fail");
    assert_eq!(error.to_string(), "h3_generation_failed");
    assert!(
        error
            .downcast_ref::<imagine_server::engine::H3Failure>()
            .is_none()
    );
    engine.stop().await.unwrap();
}
