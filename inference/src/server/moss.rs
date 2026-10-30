use super::*;
use crate::components as c;
/// `images` are reference images to edit with (Moss-Image-Edit), RGBA in [0, 1];
/// `maps` drive the color/depth control adapter.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate(
    rt: &Runtime,
    aimdo: &Aimdo,
    r: &Request,
    images: &[Tensor],
    maps: &ControlMaps,
    kept: &mut Kept,
    keys: Option<(ComponentKey, ComponentKey)>,
    observer: &mut Observer<'_>,
) -> Result<Tensor, Error> {
    let dev = Device::Cuda(0);
    let pins = c::HostPins::new(rt.native.clone())?;
    let mut manager = Manager::new();
    let mut memory = c::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    let (text_key, dit_key) = keys.unzip();
    // A kept diffusion model already carries its LoRA patches.
    let loras = if dit_key.is_some() && kept.dit.is_some() {
        Vec::new()
    } else {
        active_loras(&r.loras)
            .map(|l| {
                Ok((
                    crate::patching::read_lora(rt, aimdo, &l.path)?,
                    l.strength,
                    l.strength,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?
    };
    observer(Event::Phase("encoding"))?;
    // Dynamic VRAM: the text encoder's layers enter a VBAR range when they
    // run, as ComfyUI's text encoders do with it on.
    let (text, dynamic_text) = if let Some(key) = text_key {
        let params = reuse(&mut kept.text, key, || {
            observe_load("text", &r.text_encoder, observer, || {
                crate::checkpoint::dynamic::prepare_text(
                    rt,
                    aimdo.path(),
                    load(rt, aimdo, &r.text_encoder, "text", &r.model)?,
                    m::moss_component_inventory_for_layout("text", true)?,
                )
            })
        })?;
        (None, Some(params))
    } else {
        let text = observe_load("text", &r.text_encoder, observer, || {
            Ok(c::StaticModel::new(
                m::static_moss_component(
                    rt,
                    "text",
                    load(rt, aimdo, &r.text_encoder, "text", &r.model)?,
                )?
                .with_shared_pins(pins.clone()),
                rt,
                None,
            ))
        })?;
        (Some(text), None)
    };
    let patcher = text.as_ref().map(|t| t.patcher());
    let mut tokenizer = m::MossTokenizer::frozen()?;
    let mut contexts = Vec::new();
    for prompt in [&r.prompt, &r.negative_prompt] {
        let tokens = tokenizer.tokenize(prompt, &Default::default())?;
        let placed;
        let params = match (dynamic_text, &text, &patcher) {
            (Some(params), _, _) => params,
            (None, Some(text), Some(patcher)) => {
                manager
                    .load(
                        std::slice::from_ref(patcher),
                        0.,
                        None,
                        false,
                        false,
                        &mut memory,
                    )
                    .map_err(|error| load_failure("text", &r.text_encoder, observer, error))?;
                placed = text.owner.borrow().execution_parameters();
                &placed
            }
            _ => unreachable!(),
        };
        let encoder = m::Qwen35TextEncoder::from_parameters(Default::default(), params)
            .map_err(|error| load_failure("text", &r.text_encoder, observer, error))?;
        let mut exec = m::MossStaticExecution::bind(rt, params, attention(rt, r.execution)?)?;
        contexts.push(
            unsafe {
                encoder.encode_prompt(
                    rt,
                    &tokens,
                    q::TextConditioningOptions {
                        execution_device: dev,
                        intermediate_device: Device::Cpu,
                    },
                    &mut exec,
                )
            }?
            .context,
        );
    }
    manager.unload_all(dev, &mut memory)?;
    drop(patcher);
    drop(text);
    rt.native.cuda_soft_empty_cache(0)?;
    // The edit encoder encodes the references after the prompt, and
    // MOSSImageControl its maps; Flux2's latent format leaves them as encoded.
    let maps = [&maps.color, &maps.depth];
    let (reference_latents, map_latents) = if images.is_empty() && maps.iter().all(|m| m.is_none())
    {
        (Vec::new(), [None, None])
    } else {
        let vae = load_vae(rt, aimdo, r, &pins, observer)?;
        let mut encode = |pixels: &Tensor| -> Result<Tensor, Error> {
            let pixels = flux2_crop_pixels(&pixels.f_narrow(-1, 0, 3)?)?.f_movedim_int(-1, 1)?;
            flux2_vae(
                rt,
                &vae,
                q::VaeOperation::Encode,
                &pixels,
                &mut manager,
                &mut memory,
                (&r.vae, observer),
                attention(rt, r.execution)?,
            )
        };
        let latents = images
            .iter()
            .map(&mut encode)
            .collect::<Result<Vec<_>, _>>()?;
        let [color, depth] = maps.map(|m| m.as_ref().map(&mut encode).transpose());
        let map_latents = [color?, depth?];
        manager.unload_all(dev, &mut memory)?;
        drop(vae);
        rt.native.cuda_soft_empty_cache(0)?;
        (latents, map_latents)
    };
    let control_latent = m::moss_control_latent(
        map_latents[0].as_ref(),
        map_latents[1].as_ref(),
        r.control.strength,
    )?;
    // Each condition carries the references, so the original counts their tokens
    // once per condition, and once in the minimum.
    let reference_shape = [
        1,
        128,
        reference_latents
            .iter()
            .map(|l| l.size()[2] * l.size()[3])
            .sum::<i64>(),
    ];
    let reference_memory = if reference_latents.is_empty() {
        0.
    } else {
        m::moss_memory_required(&reference_shape, Kind::BFloat16, true)
    };
    observer(Event::Phase("loading_model"))?;
    let shape = [1, 128, r.height / 16, r.width / 16];
    let mut placed = None;
    let (model, patcher, params) = if let Some(key) = dit_key {
        // Dynamic VRAM: layers enter a VBAR range when they run, as ComfyUI.
        let params = reuse(&mut kept.dit, key, || {
            observe_load("dit", &r.diffusion_model, observer, || {
                let mut params = load(rt, aimdo, &r.diffusion_model, "dit", &r.model)?;
                params.values.insert(
                    "model_sampling.sigmas".into(),
                    crate::execution_value::Value::tensor(m::MossSampling::moss()?.sigmas),
                );
                let fused = params.contains_key("layers.0.proj.qkv.weight");
                let inventory = m::moss_component_inventory_for_layout("dit", fused)?;
                if !loras.is_empty() {
                    let patches = m::component_patch_owner(&params, &loras)?;
                    bind_dynamic_patches(&patches, &mut params, &inventory.modules)?;
                }
                crate::checkpoint::dynamic::prepare_model(rt, aimdo.path(), params, inventory)
            })
        })?;
        (None, None, params)
    } else {
        let model = observe_load("dit", &r.diffusion_model, observer, || {
            let params = load(rt, aimdo, &r.diffusion_model, "dit", &r.model)?;
            let patches = if loras.is_empty() {
                None
            } else {
                Some(m::component_patch_owner(&params, &loras)?)
            };
            let component =
                m::static_moss_component(rt, "dit", params)?.with_shared_pins(pins.clone());
            let component = if let Some(patches) = patches {
                component.with_patches(patches)?
            } else {
                component
            };
            Ok(c::StaticModel::new(
                component,
                rt,
                Some(tch::Kind::BFloat16),
            ))
        })?;
        let patcher = model.patcher();
        manager
            .load(
                std::slice::from_ref(&patcher),
                0.,
                None,
                false,
                false,
                &mut memory,
            )
            .map_err(|error| load_failure("dit", &r.diffusion_model, observer, error))?;
        let mut doubled = shape;
        doubled[0] *= 2;
        manager
            .load(
                std::slice::from_ref(&patcher),
                m::moss_memory_required(&doubled, Kind::BFloat16, true) + 2. * reference_memory,
                Some(m::moss_memory_required(&shape, Kind::BFloat16, true) + reference_memory),
                false,
                false,
                &mut memory,
            )
            .map_err(|error| load_failure("dit", &r.diffusion_model, observer, error))?;
        let params = &*placed.insert(model.owner.borrow().execution_parameters());
        (Some(model), Some(patcher), params)
    };
    let latent = {
        let fused = params.contains_key("layers.0.proj.qkv.weight");
        let dit = m::MossDiT::from_parameters(
            m::MossConfig {
                fused_qkv: fused,
                ..Default::default()
            },
            params,
        )?;
        let sampling = m::MossSampling {
            shift: 5.,
            multiplier: 1.,
            sigmas: params["model_sampling.sigmas"].as_tensor()?,
        };
        let sigmas = if r.sampling.custom_sigmas.is_empty() {
            sampling.normal(
                r.sampling.steps,
                r.sampling.scheduler == "sgm_uniform",
                &rt.native,
            )?
        } else {
            Tensor::f_from_slice(&r.sampling.custom_sigmas)?
        }
        .f_to_device(dev)?;
        // A control adapter widens the video embedder; without maps its extra
        // channels are zeros, as MOSSImage.concat_cond gives.
        let embedder = match &params["video_embedder.weight"] {
            value @ crate::execution_value::Value::Quantized(_) => {
                value.as_quantized()?.params.original_shape[1]
            }
            value => value.as_tensor()?.size()[1],
        };
        let control = m::MossControl::new(
            control_latent.as_ref(),
            embedder,
            &shape,
            dev,
            (r.control.start, r.control.end),
            &sampling,
        )?;
        if control_latent.is_some() && control.is_none() {
            return Err(Error::Native(
                "Control maps need the color/depth control adapter".into(),
            ));
        }
        let mut rng = rt.native.generator(Device::Cpu, r.seed)?;
        let noise = rng.randn(&shape, Kind::Float)?.f_to_device(dev)?;
        let input = q::flow_noise_scaling(
            &sigmas.f_get(0)?,
            &noise,
            &Tensor::f_zeros(shape, (Kind::Float, dev))?,
            1.,
        )?;
        // Keys and values of the references, computed once for both conditions.
        let reference_kv = if reference_latents.is_empty() {
            None
        } else {
            let refs = reference_latents
                .iter()
                .map(|l| l.f_to_device_(dev, Kind::BFloat16, false, false))
                .collect::<Result<Vec<_>, _>>()?;
            let mut exec = m::MossStaticExecution::bind(rt, params, attention(rt, r.execution)?)?;
            Some(unsafe { dit.preprocess_refs(rt, &refs, &mut exec)? })
        };
        let condition = |context| q::FullAreaCondition {
            context,
            attention_mask: None,
            reference_latents: None,
            image_slots: None,
            concat: None,
        };
        let negative = condition(contexts.pop().unwrap());
        let positive = condition(contexts.pop().unwrap());
        observer(Event::Phase("inference"))?;
        let mut preview = super::preview_callback::State::new(r.sampling_preview);
        let mut denoiser = unsafe {
            m::MossDenoiser::bind(m::MossDenoiserConfig {
                runtime: rt,
                model: &dit,
                positive,
                negative: Some(negative),
                cfg: r.sampling.cfg_scale,
                compute_dtype: Kind::BFloat16,
                flash_attention_memory: true,
                multiplier: 1.,
                execution: m::MossStaticExecution::bind(rt, params, attention(rt, r.execution)?)?,
                generator: rt.native.generator(dev, r.seed)?,
                observer: Some(Box::new(|step| {
                    observer(Event::Step(step.i + 1))?;
                    preview.update(&r.model, step.i + 1, step.denoised, observer)
                })),
                references: reference_kv.as_ref().map(|kv| m::MossReferences {
                    kv,
                    layout: m::MossEditLayout::TextStart,
                }),
                rope: Default::default(),
                control,
            })
        };
        let out = q::sample_euler(input, &sigmas, Default::default(), &mut denoiser)?;
        q::flow_inverse_noise_scaling(&sigmas.f_get(sigmas.size()[0] - 1)?, &out)?
            .f_to_device(Device::Cpu)?
    };
    if model.is_some() {
        manager.unload_all(dev, &mut memory)?;
    }
    drop(patcher);
    drop(model);
    drop(placed);
    rt.native.cuda_soft_empty_cache(0)?;
    observer(Event::Phase("saving_output"))?;
    let vae = load_vae(rt, aimdo, r, &pins, observer)?;
    let out = flux2_vae(
        rt,
        &vae,
        q::VaeOperation::Decode,
        &latent,
        &mut manager,
        &mut memory,
        (&r.vae, observer),
        attention(rt, r.execution)?,
    )?;
    manager.unload_all(dev, &mut memory)?;
    Ok(out)
}
fn load_vae<'a>(
    rt: &'a Runtime,
    aimdo: &Aimdo,
    r: &Request,
    pins: &c::HostPins,
    observer: &mut Observer<'_>,
) -> Result<std::rc::Rc<c::StaticModel<'a, c::ComponentBuffers>>, Error> {
    observe_load("vae", &r.vae, observer, || {
        Ok(c::StaticModel::new(
            m::static_moss_component(rt, "vae", load(rt, aimdo, &r.vae, "vae", &r.model)?)?
                .with_shared_pins(pins.clone()),
            rt,
            None,
        ))
    })
}
/// One Flux2 VAE encode or decode under the original memory policy.
#[allow(clippy::too_many_arguments)]
fn flux2_vae<'a>(
    rt: &'a Runtime,
    model: &std::rc::Rc<c::StaticModel<'a, c::ComponentBuffers>>,
    op: q::VaeOperation,
    input: &Tensor,
    manager: &mut Manager<'a>,
    memory: &mut c::StaticMemoryServices,
    diagnostic: (&std::path::Path, &mut Observer<'_>),
    attention: Attention,
) -> Result<Tensor, Error> {
    let patcher = model.patcher();
    manager
        .load(
            std::slice::from_ref(&patcher),
            0.,
            None,
            false,
            false,
            memory,
        )
        .map_err(|error| load_failure("vae", diagnostic.0, diagnostic.1, error))?;
    let mut exec = m::MossStaticExecution::bind(rt, &PreparedParameters::default(), attention)?;
    let mut services = m::ManagedFlux2Vae {
        runtime: rt,
        model: model.clone(),
        patcher,
        manager,
        memory,
        execution: &mut exec,
        config: Default::default(),
        device: Device::Cuda(0),
        force_full_load: false,
    };
    // 426f5eea runs the Flux2 VAE in fp16, its first working dtype, where served.
    q::ImageVaePolicy {
        device: Device::Cuda(0),
        dtype: Kind::Half,
        output_device: Device::Cpu,
        output_kind: Kind::Float,
        default_kind: Kind::Float,
        output_channels: 3,
        latent_channels: 128,
    }
    .execute_with_memory(
        op,
        input,
        &mut services,
        q::ImageVaeMemory::Flux2 {
            amd: false,
            fused: true,
        },
    )
}
/// The original VAE.vae_encode_crop_pixels for Flux2: a centered crop of each
/// spatial dimension to the 16-pixel compression, keeping the RGB channels.
fn flux2_crop_pixels(pixels: &Tensor) -> Result<Tensor, Error> {
    let shape = pixels.size();
    let mut out = pixels.shallow_clone();
    for (axis, &dim) in shape.iter().enumerate().take(shape.len() - 1).skip(1) {
        let size = dim / 16 * 16;
        if size != dim {
            out = out.f_narrow(axis as i64, dim % 16 / 2, size)?;
        }
    }
    Ok(out)
}
