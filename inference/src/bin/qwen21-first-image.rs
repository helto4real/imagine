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
fn record_lora(
    own: &Path,
    component: &str,
    owner: &imagine_inference::patching::PatchOwner,
) -> Result<()> {
    let observations=owner.patches.iter().map(|(key,patches)|(format!("{}{key}",owner.key_prefix),patches.iter().map(|p|json!({"strength":p.strength,"model_strength":p.model_strength,"offset":p.offset})).collect::<Vec<_>>())).collect::<BTreeMap<_,_>>();
    fs::write(
        own.join(format!("lora-{component}.json")),
        serde_json::to_vec_pretty(&observations)?,
    )?;
    Ok(())
}
fn validate_profile(p: &Value) -> Result<()> {
    let registered = [
        include_str!("../../profiles/qwen21-first-int8.json"),
        include_str!("../../profiles/qwen21-1600x896-int8.json"),
        include_str!("../../profiles/qwen21-1600x896-40steps-int8.json"),
        include_str!("../../profiles/qwen21-1600x896-40steps-bf16.json"),
        include_str!("../../profiles/qwen21-1600x896-40steps-bf16-offload.json"),
        include_str!("../../profiles/qwen21-1600x896-8steps-turbo-offload.json"),
        include_str!("../../profiles/qwen21-editing-int8.json"),
        include_str!("../../profiles/qwen21-editing-bf16.json"),
        include_str!("../../profiles/qwen21-editing-turbo.json"),
        include_str!("../../profiles/qwen21-editing-int8-two-references.json"),
        include_str!("../../profiles/qwen21-editing-int8-keep-vision.json"),
        include_str!("../../profiles/qwen21-portrait-1mpx-int8.json"),
        include_str!("../../profiles/qwen21-editing-portrait-1mpx-int8.json"),
        include_str!("../../profiles/qwen21-editing-portrait-1mpx-int8-seed3061383787.json"),
        include_str!("../../profiles/qwen21-lora-int8.json"),
        include_str!("../../profiles/qwen21-lora-bf16.json"),
        include_str!("../../profiles/qwen21-lora-turbo.json"),
        include_str!("../../profiles/qwen21-lora-text-int8.json"),
        include_str!("../../profiles/qwen21-lora-stack-int8.json"),
        include_str!("../../profiles/qwen21-lora-portrait-1600-int8.json"),
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
    reference_latents: Vec<Tensor>,
    image_slots: Vec<i64>,
    cache: q::PrefixCache<'a, q::StaticCacheServices>,
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
                &self.reference_latents,
                &self.image_slots,
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
// Explicit development diagnostic: replay recorded original placements to
// distinguish arithmetic differences from different physical allocator inputs.
// Normal runs continue to use the original automatic load policy.
fn replay_memory_plan(
    rt: &Runtime,
    owner: &mut q::StaticOffload,
    own: &Path,
    profile: &Value,
    phase: &str,
) -> Result<()> {
    let path = own.join("reference-memory-plan.json");
    if !path.is_file() {
        return Ok(());
    }
    let plan: Value = serde_json::from_slice(&fs::read(path)?)?;
    if plan["profile"] != *profile {
        return Err("Recorded memory diagnostic profile differs".into());
    }
    let record = plan["phases"]
        .as_array()
        .ok_or("Missing diagnostic phases")?
        .iter()
        .find(|p| p["phase"] == phase)
        .ok_or("Missing diagnostic phase")?;
    let placement = q::StaticPlacement {
        resident: record["resident"]
            .as_array()
            .ok_or("Missing diagnostic residency")?
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect(),
        loaded_bytes: record["loaded_bytes"].as_i64().unwrap(),
        buffer_bytes: record["buffer_bytes"].as_u64().unwrap(),
        lowvram_passes: serde_json::from_value(
            record
                .get("lowvram_passes")
                .ok_or("Recorded diagnostic lacks original lowvram function state")?
                .clone(),
        )?,
    };
    let sampling_device = match record["sampler_buffer"]["device"].as_str() {
        Some("cpu") => Device::Cpu,
        Some("cuda:0") => Device::Cuda(0),
        _ => return Err("Recorded diagnostic lacks sampler buffer placement".into()),
    };
    let lowvram = record["model_lowvram"]
        .as_bool()
        .ok_or("Recorded diagnostic lacks lowvram state")?;
    owner.replay_state(rt, placement, sampling_device, lowvram)?;
    Ok(())
}
fn memory_record(output: &Path, phase: &str, owner: &q::StaticOffload) -> Result<Value> {
    save_f32(
        &owner.sampling.sigmas,
        &output.join(format!("sampler-{phase}-f32.bin")),
    )?;
    Ok(
        json!({"phase":phase,"placement":owner.placement,"pinned_bytes":owner.pinned_bytes,
        "model_size":owner.model_size(),"model_lowvram":owner.model_lowvram,
        "sampler_buffer":{"device":if owner.sampling.sigmas.device()==Device::Cpu {"cpu"} else {"cuda:0"},
            "dtype":"float32","shape":owner.sampling.sigmas.size(),"bytes":owner.sampling.buffer_bytes()}}),
    )
}
fn generate(
    rt: &Runtime,
    aimdo: &Aimdo,
    paths: &BTreeMap<String, PathBuf>,
    model_root: &Path,
    p: &Value,
    own: &Path,
) -> Result<()> {
    // This function is reached only for an explicit --run invocation.
    let output = own.join("first-image");
    fs::create_dir(&output)?;
    let total_start = Instant::now();
    let mut loras = Vec::new();
    for spec in p["lora"].as_array().unwrap() {
        let relative = Path::new(spec["path"].as_str().ok_or("Missing LoRA path")?);
        if relative.is_absolute()
            || relative
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err("Invalid registered LoRA path".into());
        }
        let path = model_root.join(relative);
        if hash_file(&path)? != spec["sha256"].as_str().ok_or("Missing LoRA hash")? {
            return Err("Registered LoRA hash changed".into());
        }
        let state = Safetensors::load_aimdo(&path, aimdo, &rt.native)?
            .tensors
            .into_iter()
            .map(|(n, t)| {
                (
                    n,
                    imagine_inference::execution_value::Value::tensor(t.tensor),
                )
            })
            .collect();
        loras.push((
            state,
            spec["strength_model"].as_f64().unwrap(),
            spec["strength_clip"].as_f64().unwrap(),
        ));
    }

    let mut timings = BTreeMap::new();
    let offload = p["source_configuration"]["force_full_load_per_component"] == false;
    let mut memory_records = Vec::new();
    let mut manager_records = Vec::new();
    let pins = q::HostPins::new(rt.native.clone())?;
    let mut memory_manager = imagine_inference::memory::Manager::new();
    let mut memory_services = q::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    let dev = Device::Cuda(0);
    let editing = p.get("reference_resolution").is_some();
    let mut references = None;
    if editing {
        let pillow = unsafe { q::Pillow::open(&required("IMAGINE_PILLOW_LIBRARY")?) }?;
        let images = p["references"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let shape = spec["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_i64().unwrap())
                    .collect::<Vec<_>>();
                let values = if let Some(hash) = spec.get("tensor_sha256") {
                    // The host prepares a selected NHWC Float32 fixture, just as
                    // the inference API receives image tensors. Only this file
                    // is mounted; image codecs and production paths are absent.
                    let bytes = fs::read(required(&format!("IMAGINE_QWEN_REFERENCE_{index}"))?)?;
                    if bytes.len() != shape.iter().product::<i64>() as usize * 4
                        || format!("{:x}", Sha256::digest(&bytes)) != hash.as_str().unwrap()
                    {
                        return Err("Registered reference tensor size/hash mismatch".into());
                    }
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                } else {
                    let multiplier = spec["byte_multiplier"].as_i64().unwrap();
                    let offset = spec["byte_offset"].as_i64().unwrap();
                    (0..shape.iter().product::<i64>())
                        .map(|i| ((i * multiplier + offset) % 256) as f32 / 255_f32)
                        .collect::<Vec<_>>()
                };
                Ok(Tensor::f_from_slice(&values)?.f_view(shape.as_slice())?)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut reference_vae = None;
        let mut encode = |pixels: &Tensor| -> std::result::Result<Tensor, Error> {
            let start = Instant::now();
            if reference_vae.is_none() {
                let params = load(rt, aimdo, "vae", &paths["vae"], dev)
                    .map_err(|e| Error::Native(e.to_string()))?;
                let schema = q::constructor_schema()?;
                reference_vae = Some(q::ImageVae::new(
                    q::ImageVaeConfig {
                        dim: schema.vae_config.dim,
                        dec_dim: schema.vae_config.dec_dim,
                        ..Default::default()
                    },
                    params,
                ));
            }
            let vae = reference_vae.as_ref().unwrap();
            let mut exec =
                q::ResidentExecution::bind(rt, &PreparedParameters::default(), 0, attention(dev))?;
            let out = vae
                .encode(rt, &q::vae_encode_pixels(pixels, dev)?, &mut exec)?
                .f_to_device(Device::Cpu)?
                .f_to_kind(Kind::Float)?;
            let mut samples = Tensor::f_empty(out.size(), (Kind::Float, Device::Cpu))?;
            samples.f_copy_(&out)?;
            rt.native.synchronize_device(0)?;
            *timings.entry("reference_vae_load_and_encode").or_insert(0.) +=
                start.elapsed().as_secs_f64();
            Ok(samples)
        };
        let prepared = q::prepare_reference_images(
            &pillow,
            &images,
            p["reference_resolution"].as_i64().unwrap(),
            if p["reference_vae"] == true {
                Some(&mut encode)
            } else {
                None
            },
        )?;
        drop(reference_vae);
        if (prepared.width, prepared.height)
            != (p["width"].as_i64().unwrap(), p["height"].as_i64().unwrap())
        {
            return Err("Source-derived reference geometry differs from registered profile".into());
        }
        for (i, image) in prepared.vision.iter().enumerate() {
            save_f32(image, &output.join(format!("reference-vision-{i}-f32.bin")))?;
        }
        for (i, latent) in prepared.reference_latents.iter().enumerate() {
            save_f32(
                latent,
                &output.join(format!("reference-latent-{i}-f32.bin")),
            )?;
        }
        references = Some(prepared);
        rt.native.cuda_soft_empty_cache(0)?;
    }
    let mut tokenizer = q::QwenTokenizer::frozen()?;
    let token_options = q::TokenizerOptions {
        images: references.as_ref().map_or(0, |r| r.vision.len()),
        prevent_empty_text: editing,
        keep_vision: references
            .as_ref()
            .is_some_and(|r| r.reference_latents.is_empty()),
        ..Default::default()
    };
    let prompt = tokenizer.tokenize_prompt(
        p["synthetic_prompt"]
            .as_str()
            .ok_or("Missing synthetic prompt")?,
        &token_options,
    )?;
    let conditioning = {
        eprintln!("loading text component");
        let start = Instant::now();
        let mut params = load(rt, aimdo, "text", &paths["text"], dev)?;
        let mut patches = q::component_patch_owner("text", &params, &loras)?;
        record_lora(&output, "text", &patches)?;
        unsafe {
            patches.apply(rt, &mut params, dev, Kind::Half)?;
        }
        let encoder = q::QwenTextEncoder::from_parameters(Default::default(), &params)?;
        let vision = if editing {
            Some(q::QwenVisionEncoder::from_parameters(
                Default::default(),
                &params,
            )?)
        } else {
            None
        };
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
            encoder.encode_prompt_with_images(
                rt,
                &prompt,
                q::TextConditioningOptions {
                    execution_device: dev,
                    intermediate_device: Device::Cpu,
                },
                vision
                    .as_ref()
                    .zip(references.as_ref())
                    .map(|(v, r)| (v, r.vision.as_slice())),
                &mut exec,
            )
        }?;
        timings.insert("text_encode", elapsed(rt, start)?);
        if editing {
            let negative = tokenizer
                .tokenize_prompt(p["negative_prompt"].as_str().unwrap(), &token_options)?;
            let negative = unsafe {
                encoder.encode_prompt_with_images(
                    rt,
                    &negative,
                    q::TextConditioningOptions {
                        execution_device: dev,
                        intermediate_device: Device::Cpu,
                    },
                    vision
                        .as_ref()
                        .zip(references.as_ref())
                        .map(|(v, r)| (v, r.vision.as_slice())),
                    &mut exec,
                )
            }?;
            save_f32(&negative.context, &output.join("negative-context-f32.bin"))?;
            fs::write(
                output.join("conditioning.json"),
                serde_json::to_vec_pretty(
                    &json!({"positive_slots":conditioning.image_slots,"negative_slots":negative.image_slots,"positive_shape":conditioning.context.size(),"negative_shape":negative.context.size(),"negative_attention_mask":negative.attention_mask.is_some()}),
                )?,
            )?;
            if negative.attention_mask.is_some() {
                return Err("Unexpected negative mask in registered editing profile".into());
            }
        }
        if editing {
            timings.insert("text_encode", elapsed(rt, start)?);
        }
        if conditioning.attention_mask.is_some()
            || (!editing && !conditioning.image_slots.is_empty())
        {
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
        let managed = if offload {
            let params = load(rt, aimdo, "dit", &paths["dit"], Device::Cpu)?;
            let patches = q::component_patch_owner("dit", &params, &loras)?;
            record_lora(&output, "dit", &patches)?;
            let owner = q::StaticOffload::new(params, rt.native.clone())?
                .with_shared_pins(pins.clone())
                .with_patches(patches)?;
            Some(q::StaticModel::new(owner, rt))
        } else {
            None
        };
        let patcher = managed.as_ref().map(|model| model.patcher());
        if let (Some(model), Some(patcher)) = (&managed, &patcher) {
            memory_manager.load(
                std::slice::from_ref(patcher),
                0.,
                None,
                false,
                false,
                &mut memory_services,
            )?;
            let mut owner = model.owner.borrow_mut();
            replay_memory_plan(rt, &mut owner, own, p, "dit_initial")?;
            memory_records.push(memory_record(&output, "dit_initial", &owner)?);
            manager_records.push(json!({"phase":"dit_initial","registered_models":memory_manager.registrations(),"total_pinned_bytes":pins.bytes()}));
        }
        let mut resident_sampling = if offload {
            None
        } else {
            Some(q::FluxSampling::qwen21()?)
        };
        let resident = if offload {
            None
        } else {
            Some({
                let mut params = load(rt, aimdo, "dit", &paths["dit"], dev)?;
                let mut patches = q::component_patch_owner("dit", &params, &loras)?;
                record_lora(&output, "dit", &patches)?;
                unsafe {
                    patches.apply(rt, &mut params, dev, Kind::Half)?;
                }
                resident_sampling.as_mut().unwrap().to_device(dev)?;
                params
            })
        };
        timings.insert("dit_load", elapsed(rt, start)?);
        let start = Instant::now();
        if let (Some(model), Some(patcher)) = (&managed, &patcher) {
            let refs = references
                .as_ref()
                .map(|r| {
                    r.reference_latents
                        .iter()
                        .map(Tensor::size)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let groups = if editing {
                vec![refs.clone(), refs]
            } else {
                vec![]
            };
            let (required, minimum) = q::qwen_sampling_memory_with_references(&shape, &groups);
            memory_manager.load(
                std::slice::from_ref(patcher),
                required,
                Some(minimum),
                false,
                false,
                &mut memory_services,
            )?;
            let mut owner = model.owner.borrow_mut();
            replay_memory_plan(rt, &mut owner, own, p, "dit_sampling")?;
            memory_records.push(memory_record(&output, "dit_sampling", &owner)?);
            manager_records.push(json!({"phase":"dit_sampling","registered_models":memory_manager.registrations(),"total_pinned_bytes":pins.bytes()}));
        }
        let borrowed = managed.as_ref().map(|model| model.owner.borrow());
        let params = borrowed
            .as_ref()
            .map(|m| &m.parameters)
            .or(resident.as_ref())
            .unwrap();
        let mut model = q::DiT::from_parameters(Default::default(), params)?;
        let mut cache = q::PrefixCache::new(
            rt,
            Default::default(),
            q::StaticCacheServices::with_shared_pins(rt.native.clone(), pins.clone())?,
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
            borrowed
                .as_ref()
                .map(|owner| &owner.sampling)
                .or(resident_sampling.as_ref())
                .unwrap()
                .simple(p["steps"].as_u64().unwrap() as usize)?
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
            reference_latents: references
                .as_ref()
                .map(|r| {
                    r.reference_latents
                        .iter()
                        .map(|latent| {
                            q::LatentNormalization::new(rt)?
                                .process_in(latent)?
                                .f_to_device_(dev, Kind::BFloat16, false, false)
                                .map_err(Error::from)
                        })
                        .collect::<std::result::Result<Vec<_>, Error>>()
                })
                .transpose()?
                .unwrap_or_default(),
            image_slots: conditioning.image_slots.clone(),
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
        fs::write(
            output.join("prefix-cache.json"),
            serde_json::to_vec_pretty(&json!({
                "store_device":denoiser.cache.store_device().map(|d|if d==Device::Cpu {"cpu"}else{"cuda:0"}),
                "memory_bytes":denoiser.cache.memory_bytes(),
                "pinned_bytes":denoiser.cache.services.pinned_bytes(),
                "staging_buffers":denoiser.cache.services.staging_buffers()
            }))?,
        )?;
        q::PrefixKv::free(&mut denoiser.cache)?;
        timings.insert("sampling", elapsed(rt, start)?);
        drop(denoiser);
        drop(model);
        drop(borrowed);
        if let Some(model) = &managed {
            memory_manager.unload_all(dev, &mut memory_services)?;
            memory_records.push(memory_record(
                &output,
                "dit_unloaded",
                &model.owner.borrow(),
            )?);
            manager_records.push(json!({"phase":"dit_unloaded","registered_models":memory_manager.registrations(),"total_pinned_bytes":pins.bytes()}));
            if memory_manager.registrations() != 0 || pins.bytes() != 0 {
                return Err("Managed DiT release retained registrations".into());
            }
        }
        if let Some(sampling) = &mut resident_sampling {
            sampling.to_device(Device::Cpu)?;
        }
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
        fs::write(
            output.join("memory-manager.json"),
            serde_json::to_vec_pretty(&manager_records)?,
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
            &json!({"schema":1,"pixels_shape":pixels.size(),"context_shape":context_shape,"latent_shape":latent_shape,"timings_seconds":timings,"timing_method":"monotonic wall clock, CUDA synchronized at phase boundaries; per-step callbacks before Euler update","dtype":"float32","channels":"RGBA","profile":if editing {"qwen21-editing"} else if p["variant"]=="turbo_bf16" {"qwen21-1600x896-8steps-turbo-offload"} else if offload {"qwen21-1600x896-40steps-bf16-offload"} else if p["variant"]=="bf16" {"qwen21-1600x896-40steps-bf16"} else if p["steps"]==40 {"qwen21-1600x896-40steps-int8"} else if p["width"]==1600 {"qwen21-1600x896-int8"} else {"qwen21-first-int8"},"source_profile":p,"cpu_threads":tch::get_num_threads(),"kitchen_triton_disabled":rt.backends.policy.list_backends()["triton"].disabled,"diagnostic_trace":own.join("trace-enabled").is_file(),"diagnostic_memory_plan":own.join("reference-memory-plan.json").is_file(),"production_runtime_read":false}),
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
    generate(&rt, &aimdo, &paths, &model_root, &p, &own)
}
