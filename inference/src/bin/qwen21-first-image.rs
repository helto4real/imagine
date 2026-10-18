//! Isolated development driver. Default is preflight; only --run executes a
//! trained forward. See the explicit test profile and its source configuration.
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::{Aimdo, PreparedParameters, Safetensors, bind_comfy_layouts},
    execution::Runtime,
    models::qwen_image21 as q,
    native::{Error, Generator, Kitchen},
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tch::{Device, Kind, Tensor};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn save_f32(t: &Tensor, path: &Path) -> Result<()> {
    let t = t
        .f_to_device_(Device::Cpu, Kind::Float, false, false)?
        .f_contiguous()?;
    let mut values = vec![0f32; t.numel()];
    let n = values.len();
    t.f_copy_data(&mut values, n)?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes)?;
    Ok(())
}
fn elapsed(rt: &Runtime, start: Instant) -> Result<f64> {
    rt.native.synchronize_device(0)?;
    Ok(start.elapsed().as_secs_f64())
}
fn required(name: &str) -> Result<PathBuf> {
    Ok(std::env::var_os(name)
        .ok_or_else(|| format!("Set {name} to the frozen isolated dependency"))?
        .into())
}
fn attention(device: Device) -> Attention {
    Attention {
        method: AttentionMethod::Pytorch,
        priority_wrapper: true,
        attention_precision: None,
        low_precision_attention: false,
        model_device: device,
    }
}
fn hash_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn header(path: &Path) -> Result<(Value, String)> {
    let mut f = File::open(path)?;
    let mut n = [0; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n);
    if n > 100_000_000 {
        return Err("Oversized safetensors header".into());
    }
    let mut data = vec![0; n as usize];
    f.read_exact(&mut data)?;
    Ok((
        serde_json::from_slice(&data)?,
        format!("{:x}", Sha256::digest(data)),
    ))
}
fn load(
    rt: &Runtime,
    aimdo: &Aimdo,
    component: &str,
    path: &Path,
    device: Device,
) -> Result<PreparedParameters> {
    let checkpoint = Safetensors::load_aimdo(path, aimdo, &rt.native)?;
    if checkpoint.metadata.get("_quantization_metadata").is_some() {
        return Err(
            "Legacy checkpoint metadata conversion is not yet bound in this initial driver".into(),
        );
    }
    let mut state = checkpoint
        .tensors
        .into_iter()
        .map(|(n, t)| (n, t.tensor))
        .collect::<BTreeMap<_, _>>();
    if component == "text" {
        state = q::normalize_text_keys(state);
    }
    Ok(q::prepare_initial_component(rt, component, state, device)?)
}
fn validate_profile(p: &Value) -> Result<()> {
    let registered = [
        include_str!("../../profiles/qwen21-first-int8.json"),
        include_str!("../../profiles/qwen21-1600x896-int8.json"),
        include_str!("../../profiles/qwen21-1600x896-40steps-int8.json"),
        include_str!("../../profiles/qwen21-1600x896-40steps-bf16.json"),
        include_str!("../../profiles/qwen21-1600x896-40steps-bf16-offload.json"),
        include_str!("../../profiles/qwen21-1600x896-8steps-turbo-offload.json"),
    ];
    if !registered
        .iter()
        .any(|source| serde_json::from_str::<Value>(source).is_ok_and(|frozen| p == &frozen))
    {
        return Err("This development driver requires one of its explicitly registered source-matched comparison profiles".into());
    }
    Ok(())
}
struct Denoiser<'a> {
    rt: &'a Runtime,
    model: &'a q::DiT,
    context: Tensor,
    cache: q::PrefixCache<'a, q::ResidentCacheServices>,
    exec: q::ResidentExecution,
    generator: Generator,
    step_start: Instant,
    step_seconds: Vec<f64>,
}
impl q::EulerExecution for Denoiser<'_> {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor) -> std::result::Result<Tensor, Error> {
        // Full-area CFG=1 calc_cond_batch still concatenates and aggregates.
        let input = Tensor::f_cat(&[x], 0)?;
        let time = Tensor::f_cat(&[sigma], 0)?.f_to_kind(Kind::Float)?;
        let output = unsafe {
            self.model.forward(
                self.rt,
                &input.f_to_kind(Kind::BFloat16)?,
                &time,
                &self.context,
                &[],
                &[],
                Some(&mut self.cache),
                &mut self.exec,
            )
        }?
        .f_to_kind(Kind::Float)?;
        let denoised = q::flow_denoised(sigma, &output, &input)?;
        q::single_full_condition_cfg1(x, &denoised)
    }
    fn noise_like(&mut self, x: &Tensor) -> std::result::Result<Tensor, Error> {
        self.generator.randn(&x.size(), x.kind())
    }
    fn callback(&mut self, s: q::EulerStep<'_>) -> std::result::Result<(), Error> {
        self.rt.native.synchronize_device(0)?;
        self.step_seconds
            .push(self.step_start.elapsed().as_secs_f64());
        self.step_start = Instant::now();
        eprintln!("sampling step {}", s.i + 1);
        Ok(())
    }
}
fn generate(
    rt: &Runtime,
    aimdo: &Aimdo,
    paths: &BTreeMap<String, PathBuf>,
    p: &Value,
    own: &Path,
) -> Result<()> {
    // This function is reached only for an explicit --run invocation.
    let output = own.join("first-image");
    fs::create_dir(&output)?;
    let total_start = Instant::now();
    let mut timings = BTreeMap::new();
    let offload = p["source_configuration"]["force_full_load_per_component"] == false;
    let mut memory_records = Vec::new();
    let dev = Device::Cuda(0);
    let mut tokenizer = q::QwenTokenizer::frozen()?;
    let prompt = tokenizer.tokenize_prompt(
        p["synthetic_prompt"]
            .as_str()
            .ok_or("Missing synthetic prompt")?,
        &q::TokenizerOptions::default(),
    )?;
    let conditioning = {
        eprintln!("loading text component");
        let start = Instant::now();
        let params = load(rt, aimdo, "text", &paths["text"], dev)?;
        let encoder = q::QwenTextEncoder::from_parameters(Default::default(), &params)?;
        let mut exec = q::ResidentExecution::bind(rt, &params, 0, attention(dev))?;
        if own.join("trace-enabled").is_file() {
            let directory = output.join("text-trace");
            fs::create_dir(&directory)?;
            let mut index = 0;
            exec.tensor_observer = Some(Box::new(move |name, tensor| {
                let stem = format!("{index:04}-{name}");
                index += 1;
                fs::write(directory.join(format!("{stem}.json")), serde_json::to_vec(&json!({"shape":tensor.size(),"strides":tensor.stride(),"dtype":format!("{:?}",tensor.kind())})).unwrap()).map_err(|e| Error::Native(e.to_string()))?;
                save_f32(tensor, &directory.join(format!("{stem}.bin")))
                    .map_err(|e| Error::Native(e.to_string()))
            }));
        }
        timings.insert("text_load", elapsed(rt, start)?);
        let start = Instant::now();
        let conditioning = unsafe {
            encoder.encode_prompt(
                rt,
                &prompt,
                q::TextConditioningOptions {
                    execution_device: dev,
                    intermediate_device: Device::Cpu,
                },
                &mut exec,
            )
        }?;
        timings.insert("text_encode", elapsed(rt, start)?);
        if conditioning.attention_mask.is_some() || !conditioning.image_slots.is_empty() {
            return Err(
                "Nontrivial mask/image conditioning has a separate initial-driver binding".into(),
            );
        }
        conditioning
    };
    rt.native.cuda_soft_empty_cache(0)?;
    save_f32(&conditioning.context, &output.join("context-f32.bin"))?;
    let context_shape = conditioning.context.size();
    let shape = [
        1,
        64,
        p["height"].as_i64().unwrap() / 16,
        p["width"].as_i64().unwrap() / 16,
    ];
    let latent = {
        eprintln!("loading diffusion component");
        let start = Instant::now();
        let load_start = start;
        let mut managed = if offload {
            let params = load(rt, aimdo, "dit", &paths["dit"], Device::Cpu)?;
            let mut owner = q::StaticOffload::new(params, rt.native.clone())?;
            owner.load_for(None, 0.)?;
            memory_records.push(json!({"phase":"dit_initial", "placement":owner.placement,"pinned_bytes":owner.pinned_bytes}));
            Some(owner)
        } else {
            None
        };
        let resident = if offload {
            None
        } else {
            Some(load(rt, aimdo, "dit", &paths["dit"], dev)?)
        };
        timings.insert("dit_load", elapsed(rt, start)?);
        let start = Instant::now();
        if let Some(owner) = &mut managed {
            let required = q::qwen_sampling_memory(2, shape[2], shape[3]);
            let minimum = q::qwen_sampling_memory(1, shape[2], shape[3]);
            owner.load_for(Some(minimum), required)?;
            memory_records.push(json!({"phase":"dit_sampling", "placement":owner.placement,"pinned_bytes":owner.pinned_bytes}));
        }
        let params = managed
            .as_ref()
            .map(|m| &m.parameters)
            .or(resident.as_ref())
            .unwrap();
        let mut model = q::DiT::from_parameters(Default::default(), params)?;
        let mut cache = q::PrefixCache::new(
            rt,
            Default::default(),
            q::ResidentCacheServices {
                native: rt.native.clone(),
            },
        );
        model.reset_prefix_cache(true, &mut cache)?;
        let exec = q::ResidentExecution::bind(rt, params, model.config.num_layers, attention(dev))?;
        let start = if offload {
            start
        } else {
            timings.insert("dit_load", elapsed(rt, load_start)?);
            Instant::now()
        };
        let sigmas = if let Some(values) = p.get("custom_sigmas") {
            let values = values
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            let sigmas = Tensor::f_from_slice(&values)?;
            save_f32(&sigmas, &output.join("sigmas-f32.bin"))?;
            sigmas
        } else {
            q::FluxSampling::qwen21()?.simple(p["steps"].as_u64().unwrap() as usize)?
        }
        .f_to_device(dev)?;
        let mut cpu_rng = rt
            .native
            .generator(Device::Cpu, p["seed"].as_u64().unwrap())?;
        let noise = cpu_rng.randn(&shape, Kind::Float)?.f_to_device(dev)?;
        let zero = Tensor::f_zeros(shape, (Kind::Float, dev))?;
        let x = q::flow_noise_scaling(&sigmas.f_get(0)?, &noise, &zero, 1.)?;
        let mut denoiser = Denoiser {
            rt,
            model: &model,
            context: conditioning
                .context
                .f_to_device_(dev, Kind::BFloat16, false, false)?,
            cache,
            exec,
            generator: rt.native.generator(dev, p["seed"].as_u64().unwrap())?,
            step_start: Instant::now(),
            step_seconds: Vec::new(),
        };
        let sampled = q::sample_euler(x, &sigmas, Default::default(), &mut denoiser)?;
        fs::write(
            output.join("step-seconds.json"),
            serde_json::to_vec_pretty(&denoiser.step_seconds)?,
        )?;
        let sampled =
            q::flow_inverse_noise_scaling(&sigmas.f_get(sigmas.size()[0] - 1)?, &sampled)?;
        let latent = q::LatentNormalization::new(rt)?
            .process_out(&sampled.f_to_kind(Kind::Float)?)?
            .f_to_device(Device::Cpu)?;
        q::PrefixKv::free(&mut denoiser.cache)?;
        timings.insert("sampling", elapsed(rt, start)?);
        latent
    };
    rt.native.cuda_soft_empty_cache(0)?;
    save_f32(&latent, &output.join("latent-f32.bin"))?;
    let latent_shape = latent.size();
    let pixels = {
        eprintln!("loading VAE component");
        let start = Instant::now();
        let params = load(rt, aimdo, "vae", &paths["vae"], dev)?;
        let schema = q::constructor_schema()?;
        let vae = q::ImageVae::new(
            q::ImageVaeConfig {
                dim: schema.vae_config.dim,
                dec_dim: schema.vae_config.dec_dim,
                ..Default::default()
            },
            params,
        );
        let mut exec =
            q::ResidentExecution::bind(rt, &PreparedParameters::default(), 0, attention(dev))?;
        timings.insert("vae_load", elapsed(rt, start)?);
        let start = Instant::now();
        let decoded = vae
            .decode(
                rt,
                &latent.f_to_device_(dev, Kind::BFloat16, false, false)?,
                &mut exec,
            )?
            .f_to_device_(Device::Cpu, Kind::Float, false, true)?;
        let mut out = Tensor::f_empty(decoded.size(), (Kind::Float, Device::Cpu))?;
        out.f_copy_(&decoded)?;
        q::vae_process_output(&mut out)?;
        timings.insert("vae_decode", elapsed(rt, start)?);
        out.f_movedim_int(1, -1)?
    };
    rt.native.cuda_soft_empty_cache(0)?;
    timings.insert(
        "pipeline_wall_including_unload_and_intermediate_export",
        total_start.elapsed().as_secs_f64(),
    );
    if offload {
        fs::write(
            output.join("memory-policy.json"),
            serde_json::to_vec_pretty(&memory_records)?,
        )?;
    }
    save_f32(&pixels, &output.join("image-rgba-f32.bin"))?;
    pixels.save(output.join("image-rgba-f32.ot"))?;
    let image = pixels
        .f_get(0)?
        .f_mul_scalar(255)?
        .f_to_kind(Kind::Uint8)?
        .f_permute([2, 0, 1])?;
    tch::vision::image::save(&image, output.join("image.png"))?;
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":1,"pixels_shape":pixels.size(),"context_shape":context_shape,"latent_shape":latent_shape,"timings_seconds":timings,"timing_method":"monotonic wall clock, CUDA synchronized at phase boundaries; per-step callbacks before Euler update","dtype":"float32","channels":"RGBA","profile":if p["variant"]=="turbo_bf16" {"qwen21-1600x896-8steps-turbo-offload"} else if offload {"qwen21-1600x896-40steps-bf16-offload"} else if p["variant"]=="bf16" {"qwen21-1600x896-40steps-bf16"} else if p["steps"]==40 {"qwen21-1600x896-40steps-int8"} else if p["width"]==1600 {"qwen21-1600x896-int8"} else {"qwen21-first-int8"},"source_profile":p,"cpu_threads":tch::get_num_threads(),"kitchen_triton_disabled":rt.backends.policy.list_backends()["triton"].disabled,"diagnostic_trace":own.join("trace-enabled").is_file(),"production_runtime_read":false}),
        )?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut options = BTreeMap::new();
    let mut run = false;
    let mut load_check = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-root" | "--profile" | "--own-root" => {
                let next = args
                    .get(i + 1)
                    .filter(|v| !v.starts_with("--"))
                    .ok_or("Missing path value")?;
                if options.insert(args[i].clone(), next.clone()).is_some() {
                    return Err("Duplicate path flag".into());
                }
                i += 2;
            }
            "--run" => {
                run = true;
                i += 1;
            }
            "--load-check" => {
                load_check = true;
                i += 1;
            }
            _ => return Err(format!("Unknown argument {}", args[i]).into()),
        }
    }
    if run && load_check {
        return Err("Choose load-only preflight or generation, not both".into());
    }
    let value = |flag: &str| -> Result<PathBuf> {
        Ok(options
            .get(flag)
            .ok_or_else(|| format!("Required {flag}"))?
            .into())
    };
    let model_root = value("--model-root")?;
    let profile_path = value("--profile")?;
    let own = value("--own-root")?;
    let p: Value = serde_json::from_slice(&fs::read(&profile_path)?)?;
    validate_profile(&p)?;
    fs::create_dir_all(&own)?;
    let mut paths = BTreeMap::new();
    let mut records = BTreeMap::new();
    let schema = q::constructor_schema()?;
    for (component, relative) in p["checkpoints"].as_object().unwrap() {
        let path = model_root.join(relative.as_str().unwrap());
        let (mut h, header_hash) = header(&path)?;
        if component == "text" {
            let original = h.as_object().unwrap();
            h = Value::Object(
                original
                    .iter()
                    .map(|(n, v)| {
                        let n = if let Some(s) = n.strip_prefix("model.language_model.") {
                            format!("model.{s}")
                        } else if let Some(s) = n.strip_prefix("model.visual.") {
                            format!("visual.{s}")
                        } else if let Some(s) = n.strip_prefix("lm_head.") {
                            format!("model.lm_head.{s}")
                        } else {
                            n.clone()
                        };
                        (n, v.clone())
                    })
                    .collect(),
            );
        }
        for spec in &schema.models[component].parameters {
            if h.get(&spec.name).map(|v| &v["shape"]) != Some(&json!(spec.shape)) {
                return Err(format!(
                    "Missing/alternate constructor shape: {component}/{}",
                    spec.name
                )
                .into());
            }
        }
        records.insert(component.clone(),json!({"path":relative,"file_bytes":fs::metadata(&path)?.len(),"header_sha256":header_hash,"parameters":schema.models[component].parameters.len()}));
        paths.insert(component.clone(), path);
    }
    eprintln!("preflight: native library");
    let native = unsafe { Kitchen::load(&required("IMAGINE_KITCHEN_LIBRARY")?) }?;
    eprintln!("preflight: compiler profile");
    let compiler = Arc::new(unsafe {
        Compiler::open(
            &required("IMAGINE_TRITON_COMPILER_PROFILE")?,
            &own.join("compiler-cache"),
            &own.join("compiler-work"),
            native.clone(),
        )
    }?);
    eprintln!("preflight: frozen runtime");
    let rt = Runtime::frozen(native, compiler)?;
    // comfy.quant_ops startup disables Kitchen Triton unless explicitly opted
    // in. Raw Kitchen's registry default differs from the ComfyUI application.
    rt.backends.policy.disable("triton");
    // Frozen model_management.py startup with PyTorch attention and no --fast
    // features: enable these three SDP bodies and reduced math-SDP precision.
    eprintln!("preflight: source math policy");
    let mut sdpa = rt.native.sdpa_state()?;
    sdpa[..3].copy_from_slice(&[1, 1, 1]);
    rt.native.set_sdpa_state(sdpa)?;
    let mut math = rt.native.comfy_math_state()?;
    math[0] = 1;
    math[1] = 0;
    rt.native.set_comfy_math_state(math)?;
    bind_comfy_layouts(&rt);
    let _grad = tch::no_grad_guard();
    eprintln!("preflight: AIMDO");
    let aimdo = unsafe { Aimdo::load(&required("IMAGINE_AIMDO_LIBRARY")?) }?;
    eprintln!("preflight: CUDA memory");
    let initial_memory = rt.native.cuda_memory(0)?;
    if load_check {
        for component in ["text", "dit", "vae"] {
            eprintln!("load-only preflight: {component}");
            let params = load(&rt, &aimdo, component, &paths[component], Device::Cuda(0))?;
            match component {
                "text" => {
                    let _model = q::QwenTextEncoder::from_parameters(Default::default(), &params)?;
                }
                "dit" => {
                    let _model = q::DiT::from_parameters(Default::default(), &params)?;
                }
                _ => {}
            }
            records.get_mut(component).unwrap()["resident_parameters"] = json!(params.len());
            records.get_mut(component).unwrap()["cuda_memory_while_resident"] =
                json!(rt.native.cuda_memory(0)?);
            drop(params);
            rt.native.cuda_soft_empty_cache(0)?;
            records.get_mut(component).unwrap()["sha256"] = json!(hash_file(&paths[component])?);
        }
    }
    let report = json!({"schema":1,"status":if load_check{"load_only_preflight_passed"}else{"headers_preflight_passed"},"checkpoints":records,"cuda_memory_before":initial_memory,"trained_forward_executed":false,"image_generation_executed":false,"production_runtime_read":false,"source_profile":p});
    let mut f = File::create(own.join("preflight.json"))?;
    f.write_all(&serde_json::to_vec_pretty(&report)?)?;
    f.write_all(b"\n")?;
    if !run {
        println!("Preflight passed. No trained forward or image generation executed.");
        return Ok(());
    }
    generate(&rt, &aimdo, &paths, &p, &own)
}
