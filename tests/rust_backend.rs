//! Synthetic private-pipe lifecycle only; no trained model or CUDA allocation.
use imagine_server::{
    comfy_engine::{Comfy, EngineKind},
    config::Config,
    contract::{self, ModelOptions, Sampling},
    engine::{Backend, EngineState},
    generation::ImagePass,
    storage::hash,
};
use serde_json::json;
use std::{os::unix::fs::PermissionsExt, path::Path, time::Duration};
use tokio::sync::watch;
fn request() -> ImagePass {
    ImagePass {
        resolved_models: None,
        model: contract::MODEL.into(),
        model_options: ModelOptions::default(),
        sampling: Sampling::default(),
        prompt: "synthetic".into(),
        negative_prompt: String::new(),
        width: 256,
        height: 128,
        seed: 42,
        references: vec![],
        maps: vec![],
    }
}
fn fake() -> (tempfile::TempDir, Comfy, watch::Receiver<EngineState>) {
    let dir = tempfile::tempdir().unwrap();
    let program = dir.path().join("worker");
    // This stdlib-only test double is explicitly not the native runtime validation.
    std::fs::write(&program,r##"#!/usr/bin/python3
import sys,json,struct,time
r=sys.stdin.buffer;w=sys.stdout.buffer
def read():
 n=struct.unpack('>I',r.read(4))[0];return r.read(n)
def emit(e,b=b''):
 e=json.dumps(e).encode();w.write(struct.pack('>I',len(e))+e+struct.pack('>I',len(b))+b);w.flush()
emit(dict(type='ready',protocol=1,attention='checkpoint_preferences'))
while True:
 try:h=json.loads(read());body=read()
 except Exception:break
 if h['model'] in ('moss-image-sft-int8','moss-image-sft-bf16'):
  assert all(shape==[h['height'],h['width']] for shape in h['reference_shapes'])
 maps=h.get('map_lengths',[])
 assert sum(h['reference_lengths'])+sum(maps)==len(body)
 assert len(h.get('map_shapes',[]))==len(maps)==len(h.get('map_roles',[]))
 assert all(n==shape[0]*shape[1]*4 for n,shape in zip(maps,h.get('map_shapes',[])))
 if maps:
  assert h['map_roles']==['color','depth'] and h['map_shapes']==[[128,256],[64,128]]
  assert h['control']==dict(strength=0.75,start=0.0,end=0.5) and h['reference_lengths']==[]
 assert len(h['reference_shapes'])==len(h['reference_lengths'])
 assert all(n==shape[0]*shape[1]*4 for n,shape in zip(h['reference_lengths'],h['reference_shapes']))
 emit(dict(type='phase',id=h['id'],phase='inference'))
 if h['prompt']=='hang':time.sleep(60)
 if h['prompt']=='failure':
  emit(dict(type='error',id=h['id'],code='rust_generation_failed'));break
 if h['prompt']=='load_vae':
  emit(dict(type='error',id=h['id'],code='model_component_load_failed',diagnostic=dict(component='vae',reason='invalid_checkpoint')));break
 emit(dict(type='progress',id=h['id'],completed=h['sampling']['steps'],total=h['sampling']['steps'],elapsed_seconds=1))
 channels=3 if h['model'] in ('moss-image-sft-int8','moss-image-sft-bf16') else 4
 if channels==3:
  assert len(h['loras'])==1 and h['loras'][0]['strength']==0.5
 emit(dict(type='result',id=h['id']),bytes([32,64,96,255][:channels])*h['width']*h['height'])
"##).unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
    let dep = dir.path().join("dependency");
    std::fs::write(&dep, b"synthetic dependency; never dlopen").unwrap();
    let mut environment = serde_json::Map::new();
    for key in [
        "IMAGINE_KITCHEN_LIBRARY",
        "IMAGINE_AIMDO_LIBRARY",
        "IMAGINE_PILLOW_LIBRARY",
        "IMAGINE_TRITON_COMPILER_PROFILE",
    ] {
        environment.insert(key.into(), json!(dep));
    }
    let files = std::collections::BTreeMap::from([
        (program.clone(), hash(&std::fs::read(&program).unwrap())),
        (dep.clone(), hash(&std::fs::read(&dep).unwrap())),
    ]);
    let manifest = dir.path().join("runtime.json");
    std::fs::write(&manifest,serde_json::to_vec(&json!({"schema":1,"reference_revision":"446ef4e005fc7134d53ce6869ebd77d2134cd39c","binary":program,"environment":environment,"files":files})).unwrap()).unwrap();
    let (tx, rx) = watch::channel(EngineState {
        backend: "rust".into(),
        ..Default::default()
    });
    let config = Config {
        project: dir.path().into(),
        data: dir.path().join("data"),
        bind: "127.0.0.1:0".parse().unwrap(),
        model_root: dir.path().join("models"),
        private_storage: None,
    };
    let engine = Comfy::new_rust(config, tx, &manifest).unwrap();
    (dir, engine, rx)
}
#[tokio::test]
async fn native_frames_preserve_raw_references_progress_png_and_persistent_child() {
    let (_dir, mut engine, state) = fake();
    assert_eq!(EngineKind::default(), EngineKind::Rust);
    assert!(state.borrow().pid.is_none());
    let mut r = request();
    let mut image = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgba8(256, 256)
        .write_to(&mut image, image::ImageFormat::Png)
        .unwrap();
    r.references.push(image.into_inner().into());
    let bytes = engine.generate(&r).await.unwrap();
    assert_eq!(
        image::load_from_memory(&bytes)
            .unwrap()
            .to_rgba8()
            .get_pixel(0, 0)
            .0,
        [32, 64, 96, 255]
    );
    let pid = state.borrow().pid.unwrap();
    assert_eq!(state.borrow().progress.as_ref().unwrap().completed, 40);
    r.references.clear();
    engine.generate(&r).await.unwrap();
    assert_eq!(state.borrow().pid, Some(pid));
    engine.stop().await.unwrap();
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
}

#[tokio::test]
async fn moss_int8_and_bf16_preserve_loras_and_rgb_worker_protocol() {
    let (_dir, mut engine, _) = fake();
    for model in [contract::MOSS, contract::MOSS_BF16] {
        let defaults = contract::model(model).unwrap()["defaults"].clone();
        let mut r = request();
        r.model = model.into();
        r.model_options = serde_json::from_value(defaults["model_options"].clone()).unwrap();
        r.sampling = serde_json::from_value(defaults["sampling"].clone()).unwrap();
        r.model_options
            .moss_image
            .as_mut()
            .unwrap()
            .loras
            .push(contract::Lora {
                path: "synthetic/style.safetensors".into(),
                strength: 0.5,
            });
        let bytes = engine.generate(&r).await.unwrap();
        assert_eq!(
            image::load_from_memory(&bytes)
                .unwrap()
                .to_rgb8()
                .get_pixel(0, 0)
                .0,
            [32, 64, 96]
        );
    }
    engine.stop().await.unwrap();
}
#[tokio::test]
async fn cancel_reaps_before_release_and_worker_failures_are_sanitized() {
    let (_dir, mut engine, mut state) = fake();
    let mut r = request();
    r.prompt = "hang".into();
    {
        let run = engine.generate(&r);
        tokio::pin!(run);
        tokio::select! {_=&mut run=>panic!("must wait for cancellation"),_=async{while state.borrow().phase!="inference"{state.changed().await.unwrap()}}=>{}}
    }
    let pid = state.borrow().pid.unwrap();
    engine.stop().await.unwrap();
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
    r.prompt = "failure".into();
    let error = engine.generate(&r).await.unwrap_err();
    assert_eq!(
        error
            .downcast_ref::<imagine_server::engine::EngineError>()
            .unwrap()
            .0,
        "rust_generation_failed"
    );
    engine.stop().await.unwrap();
}

#[tokio::test]
async fn rust_component_errors_are_not_replaced_with_generic_generation_failure() {
    let (_dir, mut engine, _) = fake();
    let mut r = request();
    r.prompt = "load_vae".into();
    let error = engine.generate(&r).await.unwrap_err();
    let details = error
        .downcast_ref::<imagine_server::component_error::ComponentLoadError>()
        .unwrap();
    assert_eq!(
        details.diagnostic.component,
        imagine_server::component_error::Component::Vae
    );
    assert_eq!(
        details.diagnostic.reason,
        imagine_server::component_error::Reason::InvalidCheckpoint
    );
    assert!(details.file.starts_with("vae/"));
    assert!(details.valid());
    engine.stop().await.unwrap();
}
#[tokio::test]
async fn unsupported_sampler_and_missing_runtime_reject_but_rebuild_needs_no_new_hash() {
    let (dir, mut engine, state) = fake();
    let mut r = request();
    r.sampling.sampler = "res_multistep".into();
    assert!(engine.generate(&r).await.is_err());
    assert!(state.borrow().pid.is_none());
    std::fs::write(dir.path().join("worker"), b"replaced binary").unwrap();
    assert!(imagine_server::rust_runtime::Runtime::load(&dir.path().join("runtime.json")).is_ok());
    std::fs::remove_file(dir.path().join("worker")).unwrap();
    assert!(imagine_server::rust_runtime::Runtime::load(&dir.path().join("runtime.json")).is_err());
    r.sampling.sampler = "euler".into();
    let changed = engine.generate(&r).await.unwrap_err();
    assert!(format!("{changed:#}").contains("runtime artifact missing"));
    assert!(state.borrow().pid.is_none());
    tokio::time::timeout(Duration::from_secs(2), engine.stop())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn moss_control_maps_reach_the_worker_as_raw_rgba_after_the_references() {
    let (_dir, mut engine, _) = fake();
    let defaults = contract::model(contract::MOSS).unwrap()["defaults"].clone();
    let mut r = request();
    r.model = contract::MOSS.into();
    r.model_options = serde_json::from_value(defaults["model_options"].clone()).unwrap();
    r.sampling = serde_json::from_value(defaults["sampling"].clone()).unwrap();
    r.model_options
        .moss_image
        .as_mut()
        .unwrap()
        .loras
        .push(contract::Lora {
            path: contract::MOSS_CONTROL_LORA.into(),
            strength: 0.5,
        });
    r.model_options.moss_control = Some(contract::ControlOptions {
        strength: 0.75,
        start: 0.,
        end: 0.5,
    });
    for (role, (w, h)) in [("color", (256, 128)), ("depth", (128, 64))] {
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(w, h)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        r.maps.push((role, png.into_inner().into()));
    }
    engine.generate(&r).await.unwrap();
    engine.stop().await.unwrap();
}

#[tokio::test]
async fn moss_edit_references_reach_the_worker_at_the_output_size() {
    let (_dir, mut engine, _) = fake();
    let defaults = contract::model(contract::MOSS).unwrap()["defaults"].clone();
    let mut r = request();
    r.model = contract::MOSS.into();
    r.model_options = serde_json::from_value(defaults["model_options"].clone()).unwrap();
    r.sampling = serde_json::from_value(defaults["sampling"].clone()).unwrap();
    r.model_options
        .moss_image
        .as_mut()
        .unwrap()
        .loras
        .push(contract::Lora {
            path: contract::MOSS_EDIT_LORA.into(),
            strength: 0.5,
        });
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(600, 400)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    r.references.push(png.into_inner().into());
    engine.generate(&r).await.unwrap();
    engine.stop().await.unwrap();
}
