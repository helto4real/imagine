use super::*;
pub(super) fn generate(
    rt: &Runtime,
    aimdo: &Aimdo,
    r: &Request,
    observer: &mut Observer<'_>,
) -> Result<Tensor, Error> {
    let dev = Device::Cuda(0);
    let pins = q::HostPins::new(rt.native.clone())?;
    let mut manager = Manager::new();
    let mut memory = q::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    observer(Event::Phase("encoding"))?;
    let text = observe_load("text", &r.text_encoder, observer, || {
        Ok(q::StaticModel::new(
            m::static_moss_component(rt, "text", load(rt, aimdo, &r.text_encoder, "text", true)?)?
                .with_shared_pins(pins.clone()),
            rt,
        ))
    })?;
    let patcher = text.patcher();
    let mut tokenizer = m::MossTokenizer::frozen()?;
    let mut contexts = Vec::new();
    for prompt in [&r.prompt, &r.negative_prompt] {
        let tokens = tokenizer.tokenize(prompt, &Default::default())?;
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
        let encoder = m::Qwen35TextEncoder::from_parameters(Default::default(), &params)
            .map_err(|error| load_failure("text", &r.text_encoder, observer, error))?;
        let mut exec = m::MossStaticExecution::bind(rt, &params, attention())?;
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
    observer(Event::Phase("loading_model"))?;
    let model = observe_load("dit", &r.diffusion_model, observer, || {
        Ok(q::StaticModel::new(
            m::static_moss_component(rt, "dit", load(rt, aimdo, &r.diffusion_model, "dit", true)?)?
                .with_shared_pins(pins.clone()),
            rt,
        ))
    })?;
    let patcher = model.patcher();
    let shape = [1, 128, r.height / 16, r.width / 16];
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
            m::moss_memory_required(&doubled, Kind::BFloat16, true),
            Some(m::moss_memory_required(&shape, Kind::BFloat16, true)),
            false,
            false,
            &mut memory,
        )
        .map_err(|error| load_failure("dit", &r.diffusion_model, observer, error))?;
    let latent = {
        let params = model.owner.borrow().execution_parameters();
        let fused = params.contains_key("layers.0.proj.qkv.weight");
        let dit = m::MossDiT::from_parameters(
            m::MossConfig {
                fused_qkv: fused,
                ..Default::default()
            },
            &params,
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
        let mut rng = rt.native.generator(Device::Cpu, r.seed)?;
        let noise = rng.randn(&shape, Kind::Float)?.f_to_device(dev)?;
        let input = q::flow_noise_scaling(
            &sigmas.f_get(0)?,
            &noise,
            &Tensor::f_zeros(shape, (Kind::Float, dev))?,
            1.,
        )?;
        let condition = |context| q::FullAreaCondition {
            context,
            attention_mask: None,
            reference_latents: None,
            image_slots: None,
        };
        let negative = condition(contexts.pop().unwrap());
        let positive = condition(contexts.pop().unwrap());
        observer(Event::Phase("inference"))?;
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
                execution: m::MossStaticExecution::bind(rt, &params, attention())?,
                generator: rt.native.generator(dev, r.seed)?,
                observer: Some(Box::new(|step| observer(Event::Step(step.i + 1)))),
            })
        };
        let out = q::sample_euler(input, &sigmas, Default::default(), &mut denoiser)?;
        q::flow_inverse_noise_scaling(&sigmas.f_get(sigmas.size()[0] - 1)?, &out)?
            .f_to_device(Device::Cpu)?
    };
    manager.unload_all(dev, &mut memory)?;
    drop(patcher);
    drop(model);
    rt.native.cuda_soft_empty_cache(0)?;
    observer(Event::Phase("saving_output"))?;
    let model = observe_load("vae", &r.vae, observer, || {
        Ok(q::StaticModel::new(
            m::static_moss_component(rt, "vae", load(rt, aimdo, &r.vae, "vae", true)?)?
                .with_shared_pins(pins.clone()),
            rt,
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
        .map_err(|error| load_failure("vae", &r.vae, observer, error))?;
    let mut exec = m::MossStaticExecution::bind(rt, &PreparedParameters::default(), attention())?;
    let mut services = m::ManagedFlux2Vae {
        runtime: rt,
        model: model.clone(),
        patcher,
        manager: &mut manager,
        memory: &mut memory,
        execution: &mut exec,
        config: Default::default(),
        device: dev,
        force_full_load: false,
    };
    let out = q::ImageVaePolicy {
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
    manager.unload_all(dev, &mut memory)?;
    Ok(out)
}
