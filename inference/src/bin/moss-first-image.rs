//! Isolated development control. Trained execution requires explicit --run.
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::{Aimdo, PreparedParameters, Safetensors, bind_comfy_layouts},
    execution::Runtime,
    memory::Manager,
    models::{moss_image as m, qwen_image21 as q},
    native::{Error, Kitchen},
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tch::{Device, Kind, Tensor};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn required(name: &str) -> Result<PathBuf> {
    Ok(std::env::var_os(name)
        .ok_or_else(|| format!("Missing {name}"))?
        .into())
}
fn save(t: &Tensor, path: &Path) -> Result<()> {
    let t = t
        .f_to_device_(Device::Cpu, Kind::Float, false, false)?
        .f_contiguous()?;
    let mut values = vec![0f32; t.numel()];
    let n = values.len();
    t.f_copy_data(&mut values, n)?;
    let mut file = BufWriter::new(File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}
fn hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn attention() -> Attention {
    Attention {
        method: AttentionMethod::Pytorch,
        priority_wrapper: true,
        attention_precision: None,
        low_precision_attention: false,
        model_device: Device::Cuda(0),
    }
}
fn load(rt: &Runtime, aimdo: &Aimdo, role: &str, path: &Path) -> Result<PreparedParameters> {
    let checkpoint = Safetensors::load_aimdo(path, aimdo, &rt.native)?;
    if checkpoint.metadata.get("_quantization_metadata").is_some() {
        return Err("Legacy quantization conversion requires a separate binding".into());
    }
    let state = checkpoint
        .tensors
        .into_iter()
        .map(|(name, t)| (name, t.tensor))
        .collect();
    Ok(m::prepare_moss_component(
        rt,
        role,
        state,
        Device::Cpu,
        q::ComponentLoad {
            dtype: Kind::BFloat16,
            freeze: role != "vae",
        },
    )?)
}
fn phase(rt: &Runtime, start: Instant) -> Result<f64> {
    rt.native.synchronize_device(0)?;
    Ok(start.elapsed().as_secs_f64())
}
fn condition(context: Tensor) -> q::FullAreaCondition {
    q::FullAreaCondition {
        context,
        attention_mask: None,
        reference_latents: None,
        image_slots: None,
    }
}
fn memory_record(
    phase: &str,
    owner: &q::StaticOffload<q::ComponentBuffers>,
    pinned: usize,
) -> Value {
    let mut placement = owner.placement.clone();
    placement
        .lowvram_passes
        .retain(|name, _| !placement.resident.contains(name));
    let mut casts = owner
        .modules
        .iter()
        .filter(|module| {
            owner.force_cast_weights
                || (module.cast_weights && !placement.resident.contains(&module.name))
        })
        .map(|module| module.name.clone())
        .collect::<Vec<_>>();
    casts.sort();
    let buffers = owner
        .sampling
        .values
        .iter()
        .map(|(name, value)| {
            Ok((
                name.clone(),
                if value.as_tensor()?.device() == Device::Cpu {
                    "cpu"
                } else {
                    "cuda:0"
                },
            ))
        })
        .collect::<std::result::Result<BTreeMap<_, _>, Error>>()
        .unwrap();
    json!({"phase":phase,"placement":placement,"model_size":owner.model_size(),"pinned_bytes":pinned,
        "model_lowvram":owner.model_lowvram,"cast_modules":casts,"buffers":buffers})
}
fn execute(
    rt: &Runtime,
    aimdo: &Aimdo,
    profile: &Value,
    paths: &BTreeMap<String, PathBuf>,
    own: &Path,
) -> Result<()> {
    let output = own.join("first-image");
    fs::create_dir(&output)?;
    let dev = Device::Cuda(0);
    let total = Instant::now();
    let mut times = BTreeMap::new();
    let pins = q::HostPins::new(rt.native.clone())?;
    let mut manager = Manager::new();
    let mut memory = q::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    let mut records = Vec::new();
    let mut model_hashes = BTreeMap::new();
    for (role, path) in paths {
        eprintln!("hashing selected MOSS {role} checkpoint");
        model_hashes.insert(role.clone(), hash(path)?);
    }
    eprintln!("loading MOSS text component");
    let start = Instant::now();
    let params = load(rt, aimdo, "text", &paths["text"])?;
    let text = q::StaticModel::new(
        m::static_moss_component(rt, "text", params)?.with_shared_pins(pins.clone()),
        rt,
        None,
    );
    let text_patcher = text.patcher();
    manager.load(
        std::slice::from_ref(&text_patcher),
        0.,
        None,
        false,
        false,
        &mut memory,
    )?;
    times.insert("text_load", phase(rt, start)?);
    let start = Instant::now();
    let mut tokenizer = m::MossTokenizer::frozen()?;
    let mut contexts = Vec::new();
    for (index, name) in ["synthetic_prompt", "negative_prompt"].iter().enumerate() {
        let tokens = tokenizer.tokenize(
            profile[name].as_str().ok_or("Missing prompt")?,
            &Default::default(),
        )?;
        manager.load(
            std::slice::from_ref(&text_patcher),
            0.,
            None,
            false,
            false,
            &mut memory,
        )?;
        let snapshot = text.owner.borrow().execution_parameters();
        let encoder = m::Qwen35TextEncoder::from_parameters(Default::default(), &snapshot)?;
        let mut execution = m::MossStaticExecution::bind(rt, &snapshot, attention())?;
        let encoded = unsafe {
            encoder.encode_prompt(
                rt,
                &tokens,
                q::TextConditioningOptions {
                    execution_device: dev,
                    intermediate_device: Device::Cpu,
                },
                &mut execution,
            )
        }?;
        save(
            &encoded.context,
            &output.join(if index == 0 {
                "context-f32.bin"
            } else {
                "negative-context-f32.bin"
            }),
        )?;
        save(
            encoded
                .attention_mask
                .as_ref()
                .ok_or("Missing source mask")?,
            &output.join(format!("text-mask-{index}-f32.bin")),
        )?;
        contexts.push(encoded.context);
        records.push(memory_record(
            if index == 0 {
                "text_positive"
            } else {
                "text_negative"
            },
            &text.owner.borrow(),
            pins.bytes(),
        ));
    }
    times.insert("text_encode", phase(rt, start)?);
    manager.unload_all(dev, &mut memory)?;
    drop(text_patcher);
    drop(text);
    rt.native.cuda_soft_empty_cache(0)?;
    let context_shapes = contexts.iter().map(Tensor::size).collect::<Vec<_>>();
    let shape = [
        1,
        128,
        profile["height"].as_i64().unwrap() / 16,
        profile["width"].as_i64().unwrap() / 16,
    ];
    eprintln!("loading MOSS diffusion component");
    let start = Instant::now();
    let params = load(rt, aimdo, "dit", &paths["dit"])?;
    let model = q::StaticModel::new(
        m::static_moss_component(rt, "dit", params)?.with_shared_pins(pins.clone()),
        rt,
        Some(tch::Kind::BFloat16),
    );
    let patcher = model.patcher();
    manager.load(
        std::slice::from_ref(&patcher),
        0.,
        None,
        false,
        false,
        &mut memory,
    )?;
    let mut doubled = shape;
    doubled[0] *= 2;
    manager.load(
        std::slice::from_ref(&patcher),
        m::moss_memory_required(&doubled, Kind::BFloat16, true),
        Some(m::moss_memory_required(&shape, Kind::BFloat16, true)),
        false,
        false,
        &mut memory,
    )?;
    times.insert("dit_load", phase(rt, start)?);
    let start = Instant::now();
    let latent = {
        let params = model.owner.borrow().execution_parameters();
        let dit = m::MossDiT::from_parameters(
            m::MossConfig {
                fused_qkv: profile["source_configuration"]["dit_fused_qkv"]
                    .as_bool()
                    .unwrap(),
                ..Default::default()
            },
            &params,
        )?;
        let sampling = m::MossSampling {
            shift: 5.,
            multiplier: 1.,
            sigmas: params
                .get("model_sampling.sigmas")
                .ok_or("Missing sampler buffer")?
                .as_tensor()?,
        };
        let sigmas = sampling
            .normal(profile["steps"].as_i64().unwrap(), false, &rt.native)?
            .f_to_device(dev)?;
        save(&sigmas, &output.join("sigmas-f32.bin"))?;
        let seed = profile["seed"].as_u64().unwrap();
        let mut rng = rt.native.generator(Device::Cpu, seed)?;
        let noise = rng.randn(&shape, Kind::Float)?.f_to_device(dev)?;
        save(&noise, &output.join("noise-f32.bin"))?;
        let input = q::flow_noise_scaling(
            &sigmas.f_get(0)?,
            &noise,
            &Tensor::f_zeros(shape, (Kind::Float, dev))?,
            1.,
        )?;
        let negative = condition(contexts.pop().unwrap());
        let positive = condition(contexts.pop().unwrap());
        let mut denoiser = unsafe {
            m::MossDenoiser::bind(m::MossDenoiserConfig {
                runtime: rt,
                model: &dit,
                positive,
                negative: Some(negative),
                cfg: profile["cfg"].as_f64().unwrap(),
                compute_dtype: Kind::BFloat16,
                flash_attention_memory: true,
                multiplier: 1.,
                execution: m::MossStaticExecution::bind(rt, &params, attention())?,
                generator: rt.native.generator(dev, seed)?,
                observer: Some(Box::new(|step| {
                    save(
                        step.denoised,
                        &output.join(format!("step-{:03}-denoised-f32.bin", step.i)),
                    )
                    .map_err(|e| Error::Native(e.to_string()))
                })),
                references: None,
                rope: Default::default(),
            })
        };
        let sampled = q::sample_euler(input, &sigmas, Default::default(), &mut denoiser)?;
        q::flow_inverse_noise_scaling(&sigmas.f_get(sigmas.size()[0] - 1)?, &sampled)?
            .f_to_device(Device::Cpu)?
    };
    times.insert("sampling", phase(rt, start)?);
    records.push(memory_record(
        "dit_sampled",
        &model.owner.borrow(),
        pins.bytes(),
    ));
    save(&latent, &output.join("latent-f32.bin"))?;
    manager.unload_all(dev, &mut memory)?;
    drop(patcher);
    drop(model);
    rt.native.cuda_soft_empty_cache(0)?;
    eprintln!("loading MOSS Flux.2 VAE");
    let start = Instant::now();
    let params = load(rt, aimdo, "vae", &paths["vae"])?;
    let model = q::StaticModel::new(
        m::static_moss_component(rt, "vae", params)?.with_shared_pins(pins.clone()),
        rt,
        None,
    );
    let patcher = model.patcher();
    manager.load(
        std::slice::from_ref(&patcher),
        0.,
        None,
        false,
        false,
        &mut memory,
    )?;
    times.insert("vae_load", phase(rt, start)?);
    let start = Instant::now();
    let mut execution =
        m::MossStaticExecution::bind(rt, &PreparedParameters::default(), attention())?;
    let mut services = m::ManagedFlux2Vae {
        runtime: rt,
        model: model.clone(),
        patcher,
        manager: &mut manager,
        memory: &mut memory,
        execution: &mut execution,
        config: Default::default(),
        device: dev,
        force_full_load: false,
    };
    let pixels = q::ImageVaePolicy {
        device: dev,
        dtype: Kind::BFloat16,
        output_device: Device::Cpu,
        output_kind: Kind::Float,
        default_kind: Kind::Float,
        output_channels: 3,
        latent_channels: 128,
    }
    .execute_with_memory(
        q::VaeOperation::Decode,
        &latent,
        &mut services,
        q::ImageVaeMemory::Flux2 { amd: false },
    )?;
    drop(services);
    times.insert("vae_decode", phase(rt, start)?);
    records.push(memory_record(
        "vae_decoded",
        &model.owner.borrow(),
        pins.bytes(),
    ));
    manager.unload_all(dev, &mut memory)?;
    drop(model);
    rt.native.cuda_soft_empty_cache(0)?;
    times.insert(
        "pipeline_wall_including_exports_and_unload",
        total.elapsed().as_secs_f64(),
    );
    save(&pixels, &output.join("image-rgb-f32.bin"))?;
    let image = pixels
        .f_get(0)?
        .f_mul_scalar(255)?
        .f_to_kind(Kind::Uint8)?
        .f_permute([2, 0, 1])?;
    tch::vision::image::save(&image, output.join("image.png"))?;
    fs::write(
        output.join("memory-policy.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":1,"source_profile":profile,"model_sha256":model_hashes,"context_shapes":context_shapes,"latent_shape":latent.size(),"pixels_shape":pixels.size(),"timings_seconds":times,"cpu_threads":tch::get_num_threads(),"production_runtime_read":false}),
        )?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut opts = BTreeMap::new();
    let mut run = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--run" => {
                run = true;
                i += 1;
            }
            "--model-root" | "--profile" | "--own-root" => {
                let next = args
                    .get(i + 1)
                    .filter(|v| !v.starts_with("--"))
                    .ok_or("Missing path value")?;
                if opts.insert(args[i].clone(), next.clone()).is_some() {
                    return Err("Duplicate flag".into());
                }
                i += 2;
            }
            _ => return Err(format!("Unknown argument {}", args[i]).into()),
        }
    }
    let option = |flag: &str| -> Result<PathBuf> {
        Ok(opts
            .get(flag)
            .ok_or_else(|| format!("Required {flag}"))?
            .into())
    };
    let root = option("--model-root")?;
    let own = option("--own-root")?;
    let profile: Value = serde_json::from_slice(&fs::read(option("--profile")?)?)?;
    if ![
        include_str!("../../profiles/moss-first-int8.json"),
        include_str!("../../profiles/moss-1024x1024-int8.json"),
    ]
    .iter()
    .any(|source| serde_json::from_str::<Value>(source).is_ok_and(|candidate| candidate == profile))
    {
        return Err("Unregistered MOSS development control".into());
    }
    fs::create_dir_all(&own)?;
    let mut paths = BTreeMap::new();
    let mut headers = BTreeMap::new();
    let mut max_bytes = 0;
    let constructor = m::moss_constructor_schema()?;
    for (role, relative) in profile["checkpoints"].as_object().unwrap() {
        let path = root.join(relative.as_str().unwrap());
        let bytes = fs::metadata(&path)?.len();
        max_bytes = max_bytes.max(bytes);
        let mut file = File::open(&path)?;
        let mut len = [0; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        if len > 100_000_000 || len > bytes - 8 {
            return Err("Invalid checkpoint header".into());
        }
        let mut data = vec![0; len as usize];
        file.read_exact(&mut data)?;
        let header = imagine_inference::checkpoint::parse_header(&data, &path)?;
        let normalized = header
            .iter()
            .filter_map(|(name, value)| {
                let shape = value["shape"]
                    .as_array()?
                    .iter()
                    .map(Value::as_i64)
                    .collect::<Option<Vec<_>>>()?;
                let key = if role == "text" {
                    q::normalized_text_key(name)
                } else {
                    name.clone()
                };
                Some((key, shape))
            })
            .collect::<Vec<_>>();
        let normalized = if role == "vae" {
            m::flux2_vae_checkpoint_shapes(normalized)
        } else {
            normalized.into_iter().collect()
        };
        for parameter in &constructor.models[role].parameters {
            if normalized.get(&parameter.name) != Some(&parameter.shape) {
                return Err(format!(
                    "Missing/alternate constructor shape: {role}/{}",
                    parameter.name
                )
                .into());
            }
        }
        headers.insert(
            role.clone(),
            json!({"bytes":bytes,"header_sha256":format!("{:x}",Sha256::digest(data))}),
        );
        paths.insert(role.clone(), path);
    }
    let required_memory = 2 * max_bytes + 4 * 1024u64.pow(3);
    let available = q::host_available_memory()?;
    if run && available < required_memory {
        return Err(
            format!("Host RAM guard: need {required_memory}, available {available}").into(),
        );
    }
    let native = unsafe { Kitchen::load(&required("IMAGINE_KITCHEN_LIBRARY")?) }?;
    let compiler = Arc::new(unsafe {
        Compiler::open(
            &required("IMAGINE_TRITON_COMPILER_PROFILE")?,
            &own.join("compiler-cache"),
            &own.join("compiler-work"),
            native.clone(),
        )
    }?);
    let rt = Runtime::frozen(native, compiler)?;
    rt.backends.policy.disable("triton");
    bind_comfy_layouts(&rt);
    let mut sdpa = rt.native.sdpa_state()?;
    sdpa[..3].copy_from_slice(&[1, 1, 1]);
    rt.native.set_sdpa_state(sdpa)?;
    let mut math = rt.native.comfy_math_state()?;
    math[0] = 1;
    math[1] = 0;
    rt.native.set_comfy_math_state(math)?;
    let _grad = tch::no_grad_guard();
    let aimdo = unsafe { Aimdo::load(&required("IMAGINE_AIMDO_LIBRARY")?) }?;
    fs::write(
        own.join("preflight.json"),
        serde_json::to_vec_pretty(
            &json!({"profile":profile,"checkpoints":headers,"host_ram_guard":{"available":available,"required":required_memory},"cuda_memory":rt.native.cuda_memory(0)?,"trained_execution_requested":run}),
        )?,
    )?;
    if run {
        execute(&rt, &aimdo, &profile, &paths, &own)?;
    } else {
        eprintln!("MOSS preflight passed; no trained weights allocated.");
    }
    Ok(())
}
