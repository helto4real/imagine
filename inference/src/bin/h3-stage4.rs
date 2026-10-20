//! Explicit isolated H3 development worker. No server/product integration.
use imagine_inference::{
    attention::{Attention, AttentionInput, AttentionMethod},
    checkpoint::{
        Aimdo, PreparedParameters, bind_comfy_layouts,
        component::{self, StorageAnnotations},
    },
    execution::Runtime,
    execution_value::Value as V,
    memory::Manager,
    models::{
        minimax_h3::{self as h3, checkpoint::Component, video_vae::VideoExecution},
        qwen_image21::{self as q, TextEncoderExecution},
    },
    native::{Error, Kitchen},
    patching::options::Options,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use tch::{Device, Kind, Tensor};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn sha(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hash = Sha256::new();
    let mut b = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        hash.update(&b[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn save(root: &Path, name: &str, t: &Tensor) -> Result<Value> {
    let t = t.f_to_device(Device::Cpu)?.f_contiguous()?;
    let bytes = t.f_view_dtype(Kind::Uint8)?.f_reshape([-1])?;
    let mut data = vec![0u8; bytes.numel()];
    let n = data.len();
    bytes.f_copy_data(&mut data, n)?;
    fs::write(root.join(format!("{name}.bin")), &data)?;
    let dtype = match t.kind() {
        Kind::Float => "float32",
        Kind::Half => "float16",
        Kind::BFloat16 => "bfloat16",
        Kind::Int64 => "int64",
        _ => return Err("Unbound probe tensor dtype".into()),
    };
    let record = json!({"shape":t.size(),"dtype":dtype,"sha256":format!("{:x}",Sha256::digest(&data)),"bytes":data.len()});
    fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(record)
}
fn load_tensor(root: &Path, name: &str) -> Result<Tensor> {
    let v: Value = serde_json::from_slice(&fs::read(root.join(format!("{name}.json")))?)?;
    let data = fs::read(root.join(format!("{name}.bin")))?;
    if format!("{:x}", Sha256::digest(&data))
        != v["sha256"].as_str().ok_or("Missing tensor checksum")?
    {
        return Err("Own probe tensor checksum mismatch".into());
    }
    let shape: Vec<i64> = serde_json::from_value(v["shape"].clone())?;
    let kind = match v["dtype"].as_str() {
        Some("float32") => Kind::Float,
        Some("float16") => Kind::Half,
        Some("bfloat16") => Kind::BFloat16,
        Some("int64") => Kind::Int64,
        _ => return Err("Unbound own tensor dtype".into()),
    };
    Ok(Tensor::f_from_data_size(&data, &shape, kind)?)
}
struct Exec {
    native: Kitchen,
    cancelled: Arc<AtomicBool>,
    attention: Attention,
    output: PathBuf,
    events: BTreeMap<String, Value>,
}
impl TextEncoderExecution for Exec {
    fn observe(&mut self, name: &str, t: &Tensor) -> std::result::Result<(), Error> {
        if name.starts_with("h3.sampling.") {
            let file = name.replace('.', "-");
            let saved = save(&self.output, &file, t).map_err(|e| Error::Native(e.to_string()))?;
            self.events.insert(name.into(), saved);
        }
        Ok(())
    }
    fn check_interrupt(&mut self) -> std::result::Result<(), Error> {
        if self.cancelled.load(Ordering::Relaxed) {
            Err(Error::ReferenceException {
                class: "InterruptProcessingException",
                message: String::new(),
            })
        } else {
            Ok(())
        }
    }
    fn queue_begin(&mut self, _: usize, _: Device) -> std::result::Result<(), Error> {
        Ok(())
    }
    fn queue_end(&mut self, _: Device) -> std::result::Result<(), Error> {
        Ok(())
    }
    fn block(
        &mut self,
        _: usize,
        f: &mut dyn FnMut(&mut Self) -> std::result::Result<(), Error>,
    ) -> std::result::Result<(), Error> {
        f(self)
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        heads: i64,
        gqa: bool,
    ) -> std::result::Result<Tensor, Error> {
        unsafe {
            self.attention.run(
                rt,
                AttentionInput {
                    q,
                    k,
                    v,
                    heads,
                    mask,
                    skip_reshape: true,
                    skip_output_reshape: false,
                    enable_gqa: gqa,
                    scale: None,
                },
            )
        }
    }
}
impl VideoExecution for Exec {
    fn free_memory(&mut self, device: Device) -> std::result::Result<u64, Error> {
        let Device::Cuda(i) = device else {
            return q::host_available_memory();
        };
        let [free, _, active, reserved] = self.native.cuda_memory(i)?;
        Ok(free + reserved - active)
    }
}
fn execution(rt: &Runtime, output: &Path) -> Exec {
    Exec {
        native: rt.native.clone(),
        cancelled: Arc::new(AtomicBool::new(false)),
        attention: Attention {
            method: AttentionMethod::Pytorch,
            priority_wrapper: true,
            attention_precision: None,
            low_precision_attention: true,
            model_device: Device::Cuda(0),
        },
        output: output.to_owned(),
        events: BTreeMap::new(),
    }
}
fn load(
    rt: &Runtime,
    aimdo: &Aimdo,
    role: Component,
    path: &Path,
    lock: &Value,
) -> Result<PreparedParameters> {
    let entry = lock["components"]
        .as_array()
        .ok_or("Missing selected checkpoint lock")?
        .iter()
        .find(|v| v["role"] == role.name())
        .ok_or("Missing locked component")?;
    if path.metadata()?.len() != entry["file_bytes"].as_u64().ok_or("Missing locked size")?
        || sha(path)?
            != entry["sha256"]
                .as_str()
                .ok_or("Missing full checkpoint hash")?
    {
        return Err("Selected trained file identity changed".into());
    }
    let annotations = StorageAnnotations::default();
    let state =
        unsafe { component::load_aimdo(rt, aimdo, path, &Options::Atom(V::None), &annotations) }?;
    Ok(h3::checkpoint::prepare_normalized(rt, role, &state)?)
}
fn memory(owner: &q::StaticOffload<q::ComponentBuffers>, pins: usize) -> Value {
    json!({"model_size":owner.model_size(),"placement":owner.placement,"model_lowvram":owner.model_lowvram,"pinned_bytes":pins,"force_cast_weights":owner.force_cast_weights})
}
#[expect(
    clippy::too_many_arguments,
    reason = "Isolated phase inputs are explicit and reviewed by the development controller"
)]
fn phase(
    rt: &Runtime,
    aimdo: &Aimdo,
    profile: &Value,
    lock: &Value,
    model_root: &Path,
    own: &Path,
    name: &str,
    reference: bool,
    load_only: bool,
) -> Result<Value> {
    let role = match name {
        "text" => Component::Text,
        "dit" => Component::Dit,
        "reference" | "video" => Component::VideoVae,
        "audio" => Component::AudioVae,
        _ => return Err("Unknown H3 phase".into()),
    };
    let output = own.join(name);
    fs::create_dir(&output)?;
    let begin = Instant::now();
    let dev = Device::Cuda(0);
    let path = model_root.join(
        profile["checkpoints"][role.name()]
            .as_str()
            .ok_or("Missing exact checkpoint")?,
    );
    let host = q::host_available_memory()?;
    // Do not start a model-sized component if its selected physical module
    // graph alone already exceeds current available host memory.
    let inventory: Value =
        serde_json::from_str(include_str!("../../resources/h3-component-modules.json"))?;
    if host
        < inventory["models"][role.name()]["model_size"]
            .as_u64()
            .ok_or("Missing original model size")?
    {
        return Err("Insufficient available host memory for the selected component graph".into());
    }
    eprintln!("H3 phase {name}: verifying and preparing selected checkpoint");
    let params = load(rt, aimdo, role, &path, lock)?;
    let mut sampling = h3::sampling::selected_schedule(rt)?;
    let auxiliary = if role == Component::Dit {
        BTreeMap::from([(
            "model_sampling.sigmas".into(),
            V::tensor(sampling.sigmas.shallow_clone()),
        )])
    } else {
        BTreeMap::new()
    };
    let pins = q::HostPins::new(rt.native.clone())?;
    let owner = h3::checkpoint::static_component(rt, role, params, auxiliary)?
        .with_shared_pins(pins.clone());
    if owner.model_size()
        != inventory["models"][role.name()]["model_size"]
            .as_u64()
            .ok_or("Missing selected original physical size")?
    {
        return Err(
            "Loaded trained component differs from the accepted original module accounting".into(),
        );
    }
    let source_dtype = if role == Component::Dit {
        Some(Kind::BFloat16)
    } else {
        None
    };
    let model = q::StaticModel::new(owner, rt, source_dtype);
    let patcher = model.patcher();
    let mut manager = Manager::new();
    let mut services = q::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    let prepared_seconds = begin.elapsed().as_secs_f64();
    let mut report = json!({"phase":name,"reference":reference,"model_sha256":lock["components"].as_array().unwrap().iter().find(|v|v["role"]==role.name()).unwrap()["sha256"],"host_available_before":host,"prepare_seconds":prepared_seconds,"prepared":memory(&model.owner.borrow(),pins.bytes())});
    let forward: Result<()> = (|| {
        if !load_only {
            let requested = match name {
                "video" => (9.5 * 30. * 256. * 384. + 270_000_000.) * 2. * 1.03,
                "reference" => (7. * 256. * 384. + 110_000_000.) * 2. * 1.03,
                "audio" => 42_000_000. * 4. * 1.03,
                "dit" => {
                    (2. * (1. * (24. * 12. * 16. * 24. + 32. * 2. * 65.)))
                        * 2.
                        * 0.01
                        * 0.114
                        * 1048576.
                }
                _ => 0.,
            };
            manager.load(
                std::slice::from_ref(&patcher),
                requested,
                None,
                false,
                false,
                &mut services,
            )?;
            rt.native.synchronize_device(0)?;
            report["loaded"] = memory(&model.owner.borrow(), pins.bytes());
            report["cuda_loaded"] = json!(rt.native.cuda_memory(0)?);
            fs::write(
                output.join("gpu-ready.json"),
                serde_json::to_vec(&json!({"phase":name,"gpu_loaded":true}))?,
            )?;
            let inference = Instant::now();
            {
                let snapshot = model.owner.borrow().execution_parameters();
                let mut exec = execution(rt, &output);
                match name {
                    "text" => {
                        let mut tokenizer = h3::tokenizer::Tokenizer::frozen()?;
                        let images = if reference {
                            vec![V::tensor(load_tensor(own, "synthetic-reference")?)]
                        } else {
                            vec![]
                        };
                        let tokens = tokenizer.tokenize(
                            profile["synthetic_prompt"]
                                .as_str()
                                .ok_or("Missing synthetic source prompt")?,
                            &images,
                            &[],
                            false,
                        )?;
                        let entries = tokens
                            .iter()
                            .map(|e| match e.token {
                                h3::tokenizer::Token::Id(id) => json!({"id":id}),
                                _ => json!({"vision":true}),
                            })
                            .collect::<Vec<_>>();
                        fs::write(
                            output.join("tokens.json"),
                            serde_json::to_vec_pretty(&entries)?,
                        )?;
                        let encoder =
                            h3::text_encoder::TextEncoder::from_parameters(&snapshot, true)?;
                        let result =
                            unsafe { encoder.encode(rt, &tokens, dev, Device::Cpu, &mut exec) }?;
                        report["context"] = save(&output, "context", &result.context)?;
                        report["token_tags"] = save(&output, "tags", &result.token_tags)?;
                    }
                    "dit" => {
                        sampling.sigmas = snapshot["model_sampling.sigmas"].as_tensor()?;
                        let sigmas = h3::sampling::simple_sigmas(&sampling, 4)?;
                        save(&output, "sigmas", &sigmas)?;
                        let latent = Tensor::f_zeros([1, 32, 12, 16, 24], (Kind::Float, dev))?;
                        let latent = h3::streams::NestedTensor::new(
                            h3::preprocessing::empty_streams(&latent)?.into(),
                        );
                        let noise = latent.prepare_noise(42, None)?;
                        for (i, value) in noise.unbind().borrow().iter().enumerate() {
                            save(&output, &format!("noise-{i}"), &value.as_tensor()?)?;
                        }
                        let context = load_tensor(&own.join("text"), "context")?;
                        let tags = load_tensor(&own.join("text"), "tags")?;
                        let conditions = if reference {
                            vec![load_tensor(&own.join("reference"), "latent")?]
                        } else {
                            vec![]
                        };
                        let refs = if reference {
                            vec![h3::layout::ReferenceBlock {
                                kind: h3::layout::ReferenceKind::Image,
                                latent_h: 16,
                                latent_w: 24,
                                latent_t: 0,
                                ref_audio_t: 0,
                            }]
                        } else {
                            vec![]
                        };
                        let denoiser = h3::dit::Denoiser::from_parameters(
                            h3::dit::Config::selected()?,
                            &snapshot,
                        )?;
                        let result = unsafe {
                            h3::sampling::sample_euler_cfg1(
                                rt,
                                &denoiser,
                                &sampling,
                                h3::sampling::EulerRequest {
                                    latent: &latent,
                                    noise: &noise,
                                    sigmas: &sigmas,
                                    context: &context,
                                    payload: h3::dit::Payload {
                                        refs: &refs,
                                        video_conditions: &conditions,
                                        text_tags: Some(&tags),
                                        seed: 42,
                                        ..Default::default()
                                    },
                                    denoise_mask: None,
                                    kind: Kind::BFloat16,
                                },
                                &mut exec,
                            )
                        }?;
                        for (i, value) in result.unbind().borrow().iter().enumerate() {
                            save(
                                &output,
                                if i == 0 {
                                    "video-latent"
                                } else {
                                    "audio-latent"
                                },
                                &value.as_tensor()?,
                            )?;
                        }
                    }
                    "video" | "reference" => {
                        let config = h3::video_vae::Config {
                            int8_attention_available: rt.int8_attention_available(Some(dev))?,
                            ..Default::default()
                        };
                        let vae = h3::video_vae::VideoVae::new(snapshot.clone(), config);
                        if name == "reference" {
                            let image = load_tensor(own, "synthetic-reference")?
                                .f_movedim_int(-1, 1)?
                                .f_to_device_(dev, Kind::Half, false, false)?
                                .f_mul_scalar(2.)?
                                .f_sub_scalar(1.)?;
                            let latent = unsafe { vae.encode(rt, &image, dev, &mut exec) }?;
                            report["latent"] = save(&output, "latent", &latent)?;
                        } else {
                            let latent = load_tensor(&own.join("dit"), "video-latent")?
                                .f_to_device_(dev, Kind::Half, false, false)?;
                            let frames = unsafe { vae.decode(rt, &latent, None, &mut exec) }?;
                            report["frames"] = save(&output, "frames", &frames)?;
                            if frames.size() != [1, 3, 39, 256, 384] {
                                return Err("Trained video time/shape gate failed".into());
                            }
                        }
                    }
                    "audio" => {
                        let vae =
                            h3::audio_vae::AudioVae::new(snapshot.clone(), Default::default());
                        let latent =
                            load_tensor(&own.join("dit"), "audio-latent")?.f_to_device(dev)?;
                        let pcm = unsafe { vae.decode(rt, &latent, &mut exec) }?;
                        report["pcm"] = save(&output, "pcm", &pcm)?;
                        if pcm.size() != [1, 2, 52000] {
                            return Err("Trained PCM time/shape gate failed".into());
                        }
                    }
                    _ => unreachable!(),
                }
                report["observations"] = json!(exec.events);
            }
            rt.native.synchronize_device(0)?;
            report["inference_seconds"] = json!(inference.elapsed().as_secs_f64());
        }
        Ok(())
    })();
    // Release the complete graph on both inference failure and success, before
    // measuring the allocator and returning the original inference error.
    let cleanup = manager.unload_all(dev, &mut services);
    report["unloaded"] = memory(&model.owner.borrow(), pins.bytes());
    report["result"] = json!(if forward.is_ok() && cleanup.is_ok() {
        "passed"
    } else {
        "failed"
    });
    if let Err(error) = &forward {
        report["error"] = json!(error.to_string());
    }
    if let Err(error) = &cleanup {
        report["cleanup_error"] = json!(error.to_string());
    }
    drop(patcher);
    drop(model);
    drop(manager);
    drop(services);
    drop(pins);
    rt.native.synchronize_device(0)?;
    rt.native.cuda_soft_empty_cache(0)?;
    report["cuda_after_release"] = json!(rt.native.cuda_memory(0)?);
    let [allocated, reserved, peak_allocated, peak_reserved] = rt.native.cuda_allocator_stats(0)?;
    report["allocator_after_release"] = json!({"allocated":allocated,"reserved":reserved,"peak_allocated":peak_allocated,"peak_reserved":peak_reserved});
    report["seconds"] = json!(begin.elapsed().as_secs_f64());
    fs::write(
        output.join("phase.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    forward?;
    cleanup?;
    Ok(report)
}
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut values = BTreeMap::new();
    let mut reference = false;
    let mut load_only = false;
    while let Some(k) = args.next() {
        if k == "--reference" {
            reference = true;
            continue;
        }
        if k == "--load-only" {
            load_only = true;
            continue;
        }
        let key = k.into_string().map_err(|_| "UTF-8 option required")?;
        if ![
            "--profile",
            "--model-lock",
            "--model-root",
            "--own-root",
            "--phase",
        ]
        .contains(&key.as_str())
        {
            return Err("Unknown H3 worker option".into());
        }
        values.insert(
            key,
            PathBuf::from(args.next().ok_or("Missing option value")?),
        );
    }
    let required = |n: &str| -> Result<&Path> {
        values
            .get(n)
            .map(PathBuf::as_path)
            .ok_or_else(|| format!("Missing {n}").into())
    };
    let profile: Value = serde_json::from_slice(&fs::read(required("--profile")?)?)?;
    if profile != serde_json::from_str::<Value>(include_str!("../../profiles/h3-short-av.json"))? {
        return Err("Only the explicit selected H3 technical profile is accepted".into());
    }
    let lock: Value = serde_json::from_slice(&fs::read(required("--model-lock")?)?)?;
    if lock["result"] != "passed" {
        return Err("Full selected model identity gate required".into());
    }
    let library =
        std::env::var_os("IMAGINE_KITCHEN_LIBRARY").ok_or("Missing explicit native library")?;
    let rt = Runtime::frozen_partial(unsafe { Kitchen::load(Path::new(&library)) }?)?;
    bind_comfy_layouts(&rt);
    rt.backends.policy.disable("triton");
    tch::set_num_threads(2);
    let _grad = tch::no_grad_guard();
    let aimdo = unsafe {
        Aimdo::load(Path::new(
            &std::env::var_os("IMAGINE_AIMDO_LIBRARY").ok_or("Missing explicit AIMDO library")?,
        ))
    }?;
    let own = required("--own-root")?;
    let name = required("--phase")?
        .to_str()
        .ok_or("UTF-8 phase required")?;
    let result = phase(
        &rt,
        &aimdo,
        &profile,
        &lock,
        required("--model-root")?,
        own,
        name,
        reference,
        load_only,
    );
    if let Err(error) = &result {
        fs::write(
            own.join(format!("{name}-failure.json")),
            serde_json::to_vec_pretty(&json!({"phase":name,"error":error.to_string()}))?,
        )?;
    }
    result.map(|_| ())
}
