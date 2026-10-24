use super::*;
use crate::components as c;
pub(super) fn generate(
    rt: &Runtime,
    aimdo: &Aimdo,
    r: &Request,
    images: &[Tensor],
    observer: &mut Observer<'_>,
) -> Result<Tensor, Error> {
    let dev = Device::Cuda(0);
    let pins = c::HostPins::new(rt.native.clone())?;
    let mut manager = Manager::new();
    let mut memory = c::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    let loras = r
        .loras
        .iter()
        .map(|l| {
            let state = crate::patching::read_lora(rt, aimdo, &l.path)?;
            Ok((state, l.strength, l.strength))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut patch_report = crate::patching::PatchReport::default();
    observer(Event::Phase("preparing_inputs"))?;
    let references = if images.is_empty() {
        None
    } else {
        let pillow = unsafe {
            q::Pillow::open(&PathBuf::from(
                std::env::var_os("IMAGINE_PILLOW_LIBRARY")
                    .ok_or_else(|| Error::Native("Missing Pillow ABI".into()))?,
            ))
        }?;
        let vae = observe_load("vae", &r.vae, observer, || {
            Ok(c::StaticModel::new(
                q::static_component(rt, "vae", load(rt, aimdo, &r.vae, "vae", &r.model)?)?
                    .with_shared_pins(pins.clone()),
                rt,
                None,
            ))
        })?;
        let mut encode = |pixels: &Tensor| {
            qwen_vae(
                rt,
                &vae,
                &q::vae_crop_pixels(pixels)?.f_movedim_int(-1, 1)?,
                q::VaeOperation::Encode,
                &mut manager,
                &mut memory,
                (&r.vae, observer),
            )
        };
        // Server preparation already applies the shared shrink/crop geometry.
        let prepared = q::prepare_reference_images(&pillow, images, 0, Some(&mut encode))?;
        manager.unload_all(dev, &mut memory)?;
        Some(prepared)
    };
    observer(Event::Phase("encoding"))?;
    let mut tokenizer = q::QwenTokenizer::frozen()?;
    let opts = q::TokenizerOptions {
        images: images.len(),
        prevent_empty_text: true,
        keep_vision: references
            .as_ref()
            .is_none_or(|r| r.reference_latents.is_empty()),
        ..Default::default()
    };
    let text = observe_load("text", &r.text_encoder, observer, || {
        let params = load(rt, aimdo, &r.text_encoder, "text", &r.model)?;
        let patches = q::component_patch_owner("text", &params, &loras)?;
        patch_report.merge(&patches.report);
        Ok(c::StaticModel::new(
            q::static_component(rt, "text", params)?
                .with_shared_pins(pins.clone())
                .with_patches(patches)?,
            rt,
            None,
        ))
    })?;
    let patcher = text.patcher();
    let mut encode = |prompt: &str| -> Result<q::TextConditioning, Error> {
        let tokens = tokenizer.tokenize_prompt(prompt, &opts)?;
        manager
            .load(
                std::slice::from_ref(&patcher),
                0.,
                None,
                false,
                false,
                &mut memory,
            )
            .map_err(|error| load_failure("text", &r.text_encoder, observer, error))?;
        let params = text.owner.borrow().execution_parameters();
        let encoder = q::QwenTextEncoder::from_parameters(Default::default(), &params)
            .map_err(|error| load_failure("text", &r.text_encoder, observer, error))?;
        let vision = if images.is_empty() {
            None
        } else {
            Some(
                q::QwenVisionEncoder::from_parameters(Default::default(), &params)
                    .map_err(|error| load_failure("text", &r.text_encoder, observer, error))?,
            )
        };
        let mut exec = q::ResidentExecution::bind(rt, &params, 0, attention())?;
        unsafe {
            encoder.encode_prompt_with_images(
                rt,
                &tokens,
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
        }
    };
    let positive = encode(&r.prompt)?;
    // The regular negative prompt is encoded even for text-to-image. Only the
    // explicit profile option produces zero conditioning, as in ComfyUI.
    let mut negative = encode(&r.negative_prompt)?;
    if r.zero_negative_conditioning {
        negative.context = negative.context.f_zeros_like()?;
    }
    let negative = Some(negative);
    manager.unload_all(dev, &mut memory)?;
    drop(patcher);
    drop(text);
    rt.native.cuda_soft_empty_cache(0)?;
    observer(Event::Phase("loading_model"))?;
    let model = observe_load("dit", &r.diffusion_model, observer, || {
        let params = load(rt, aimdo, &r.diffusion_model, "dit", &r.model)?;
        let patches = q::component_patch_owner("dit", &params, &loras)?;
        patch_report.merge(&patches.report);
        Ok(c::StaticModel::new(
            q::StaticOffload::new(params, rt.native.clone())?
                .with_shared_pins(pins.clone())
                .with_patches(patches)?,
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
    let shape = [1, 64, r.height / 16, r.width / 16];
    let refs = references
        .as_ref()
        .map(|r| {
            r.reference_latents
                .iter()
                .map(Tensor::size)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let groups = if refs.is_empty() {
        vec![]
    } else {
        vec![refs.clone(), refs]
    };
    let (required, minimum) = q::qwen_sampling_memory_with_references(&shape, &groups);
    manager
        .load(
            std::slice::from_ref(&patcher),
            required,
            Some(minimum),
            false,
            false,
            &mut memory,
        )
        .map_err(|error| load_failure("dit", &r.diffusion_model, observer, error))?;
    let latent = {
        let owner = model.owner.borrow();
        let params = &owner.parameters;
        let sampling = &owner.sampling;
        let mut dit = q::DiT::from_parameters(Default::default(), params)?;
        let mut cache = q::PrefixCache::new(
            rt,
            Default::default(),
            q::StaticCacheServices::with_shared_pins(rt.native.clone(), pins.clone())?,
        );
        dit.reset_prefix_cache(true, &mut cache)?;
        let sigmas = if !r.sampling.custom_sigmas.is_empty() {
            Tensor::f_from_slice(&r.sampling.custom_sigmas)?
        } else {
            match r.sampling.scheduler.as_str() {
                "beta" => sampling.beta(r.sampling.steps, 0.6, 0.6, &rt.native, &mut |_| {})?,
                "simple" => sampling.simple(r.sampling.steps as usize)?,
                "normal" => sampling.normal(r.sampling.steps, false)?,
                "sgm_uniform" => sampling.normal(r.sampling.steps, true)?,
                _ => unreachable!(),
            }
        }
        .f_to_device(dev)?;
        let mut rng = rt.native.generator(Device::Cpu, r.seed)?;
        let noise = rng.randn(&shape, Kind::Float)?.f_to_device(dev)?;
        let input = q::flow_noise_scaling(
            &sigmas.f_get(0)?,
            &noise,
            &Tensor::f_zeros(shape, (Kind::Float, dev))?,
            1.,
        )?;
        observer(Event::Phase("inference"))?;
        let mut preview = super::preview_callback::State::new(r.sampling_preview);
        let mut denoiser = unsafe {
            q::QwenDenoiser::bind(q::QwenDenoiserConfig {
                runtime: rt,
                model: &dit,
                positive: full_condition(&positive, references.as_ref(), rt)?,
                negative: negative
                    .as_ref()
                    .map(|c| full_condition(c, references.as_ref(), rt))
                    .transpose()?,
                cfg: r.sampling.cfg_scale,
                disable_cfg1_optimization: false,
                flash_attention_memory: true,
                cache,
                execution: q::ResidentExecution::bind(
                    rt,
                    params,
                    dit.config.num_layers,
                    attention(),
                )?,
                compute_dtype: Kind::BFloat16,
                generator: rt.native.generator(dev, r.seed)?,
                observer: Some(Box::new(|step| {
                    observer(Event::Step(step.i + 1))?;
                    preview.update(&r.model, step.i + 1, step.denoised, observer)
                })),
            })
        };
        let out = match r.sampling.sampler.as_str() {
            "er_sde" => {
                q::sample_er_sde(input, &sigmas, sampling, Default::default(), &mut denoiser)?
            }
            "euler" => q::sample_euler(input, &sigmas, Default::default(), &mut denoiser)?,
            _ => unreachable!(),
        };
        q::PrefixKv::free(denoiser.cache_mut())?;
        q::LatentNormalization::new(rt)?
            .process_out(
                &q::flow_inverse_noise_scaling(&sigmas.f_get(sigmas.size()[0] - 1)?, &out)?
                    .f_to_kind(Kind::Float)?,
            )?
            .f_to_device(Device::Cpu)?
    };
    manager.unload_all(dev, &mut memory)?;
    drop(patcher);
    drop(model);
    rt.native.cuda_soft_empty_cache(0)?;
    observer(Event::Phase("saving_output"))?;
    let vae = observe_load("vae", &r.vae, observer, || {
        Ok(c::StaticModel::new(
            q::static_component(rt, "vae", load(rt, aimdo, &r.vae, "vae", &r.model)?)?
                .with_shared_pins(pins.clone()),
            rt,
            None,
        ))
    })?;
    let pixels = qwen_vae(
        rt,
        &vae,
        &latent,
        q::VaeOperation::Decode,
        &mut manager,
        &mut memory,
        (&r.vae, observer),
    )?;
    manager.unload_all(dev, &mut memory)?;
    Ok(pixels)
}
