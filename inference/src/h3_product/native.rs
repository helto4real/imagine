//! Frozen H3 algorithms with a new private transport adapter; one job per process.
use super::{
    deployment::{self, Config},
    protocol::{self as p, ErrorCode, Event, Phase, Request},
    wire::{self, Emit, Generator},
};
use crate::{
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
    sampling::model_sampling::ModelSampling,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io, path::PathBuf, time::Instant};
use tch::{Device, Kind, Tensor};
type NResult<T> = Result<T, Error>;
const LOCK: &[u8] = include_bytes!("../../../design/h3-stage4-evidence/selected-model-sha256.json");
const PROFILE: &str = include_str!("../../profiles/h3-short-av.json");
fn native_error() -> Error {
    Error::Native("H3 product boundary failed".into())
}
fn classify(error: &Error, fallback: ErrorCode) -> ErrorCode {
    match error.root() {
        Error::AcceleratorException { code: 2, .. }
        | Error::ReferenceException {
            class: "OutOfMemoryError",
            ..
        } => ErrorCode::OutOfMemory,
        _ => fallback,
    }
}
struct Models {
    root: PathBuf,
    lock: Value,
    profile: Value,
}
impl Models {
    fn new(root: PathBuf) -> Result<Self, ErrorCode> {
        if format!("{:x}", Sha256::digest(LOCK)) != p::MODEL_LOCK_SHA256 {
            return Err(ErrorCode::DeploymentMismatch);
        }
        Ok(Self {
            root,
            lock: serde_json::from_slice(LOCK).map_err(|_| ErrorCode::DeploymentMismatch)?,
            profile: serde_json::from_str(PROFILE).map_err(|_| ErrorCode::DeploymentMismatch)?,
        })
    }
    fn path(&self, role: Component) -> Result<PathBuf, ErrorCode> {
        let relative = self.profile["checkpoints"][role.name()]
            .as_str()
            .ok_or(ErrorCode::DeploymentMismatch)?;
        Ok(self.root.join(relative))
    }
    fn verify(&self, role: Component) -> Result<PathBuf, ErrorCode> {
        let path = self.path(role)?;
        let entry = self.lock["components"]
            .as_array()
            .ok_or(ErrorCode::DeploymentMismatch)?
            .iter()
            .find(|entry| entry["role"] == role.name())
            .ok_or(ErrorCode::DeploymentMismatch)?;
        if path
            .metadata()
            .map_err(|_| ErrorCode::ModelIdentityMismatch)?
            .len()
            != entry["file_bytes"]
                .as_u64()
                .ok_or(ErrorCode::DeploymentMismatch)?
            || deployment::sha(&path).map_err(|_| ErrorCode::ModelIdentityMismatch)?
                != entry["sha256"]
                    .as_str()
                    .ok_or(ErrorCode::DeploymentMismatch)?
        {
            return Err(ErrorCode::ModelIdentityMismatch);
        }
        Ok(path)
    }
    fn verify_all(&self) -> Result<(), ErrorCode> {
        for role in [
            Component::Text,
            Component::Dit,
            Component::VideoVae,
            Component::AudioVae,
        ] {
            self.verify(role)?;
        }
        Ok(())
    }
}
enum Notice {
    Phase(Phase),
    Step(u32),
}
struct Exec<'a> {
    native: Kitchen,
    attention: Attention,
    step: u32,
    notify: &'a mut dyn FnMut(Notice) -> NResult<()>,
}
impl TextEncoderExecution for Exec<'_> {
    fn observe(&mut self, name: &str, _: &Tensor) -> NResult<()> {
        if let Some(index) = name
            .strip_prefix("h3.sampling.")
            .and_then(|s| s.strip_suffix(".denoised"))
        {
            let step = index.parse::<u32>().map_err(|_| native_error())? + 1;
            if step > self.step {
                if step != self.step + 1 || step > p::STEPS {
                    return Err(native_error());
                }
                (self.notify)(Notice::Step(step))?;
                self.step = step;
            }
        }
        Ok(())
    }
    fn check_interrupt(&mut self) -> NResult<()> {
        Ok(())
    }
    fn queue_begin(&mut self, _: usize, _: Device) -> NResult<()> {
        Ok(())
    }
    fn queue_end(&mut self, _: Device) -> NResult<()> {
        Ok(())
    }
    fn block(&mut self, _: usize, f: &mut dyn FnMut(&mut Self) -> NResult<()>) -> NResult<()> {
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
    ) -> NResult<Tensor> {
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
impl VideoExecution for Exec<'_> {
    fn free_memory(&mut self, device: Device) -> NResult<u64> {
        let Device::Cuda(i) = device else {
            return q::host_available_memory();
        };
        let [free, _, active, reserved] = self.native.cuda_memory(i)?;
        Ok(free + reserved - active)
    }
}
fn budget(role: Component) -> f64 {
    match role {
        Component::Text => 0.,
        Component::Dit => {
            2. * (24. * 12. * 16. * 24. + 32. * 2. * 65.) * 2. * 0.01 * 0.114 * 1048576.
        }
        Component::VideoVae => (9.5 * 30. * 256. * 384. + 270_000_000.) * 2. * 1.03,
        Component::AudioVae => 42_000_000. * 4. * 1.03,
    }
}
fn component<T>(
    rt: &Runtime,
    aimdo: &Aimdo,
    models: &Models,
    role: Component,
    notify: &mut dyn FnMut(Notice) -> NResult<()>,
    forward: impl FnOnce(&PreparedParameters, &mut ModelSampling, &mut Exec<'_>) -> NResult<T>,
) -> Result<T, ErrorCode> {
    let inventory: Value =
        serde_json::from_str(include_str!("../../resources/h3-component-modules.json"))
            .map_err(|_| ErrorCode::DeploymentMismatch)?;
    let expected = inventory["models"][role.name()]["model_size"]
        .as_u64()
        .ok_or(ErrorCode::DeploymentMismatch)?;
    if q::host_available_memory().map_err(|_| ErrorCode::ComponentLoadFailed)? < expected {
        return Err(ErrorCode::OutOfMemory);
    }
    // Keep the accepted per-component full identity check as well as the
    // all-four preallocation check required by the product deployment contract.
    let path = models.verify(role)?;
    let setup = (|| -> NResult<_> {
        let annotations = StorageAnnotations::default();
        let state = unsafe {
            component::load_aimdo(rt, aimdo, &path, &Options::Atom(V::None), &annotations)
        }?;
        let parameters = h3::checkpoint::prepare_normalized(rt, role, &state)?;
        let sampling = h3::sampling::selected_schedule(rt)?;
        let auxiliary = if role == Component::Dit {
            BTreeMap::from([(
                "model_sampling.sigmas".into(),
                V::tensor(sampling.sigmas.shallow_clone()),
            )])
        } else {
            BTreeMap::new()
        };
        let pins = q::HostPins::new(rt.native.clone())?;
        let owner = h3::checkpoint::static_component(rt, role, parameters, auxiliary)?
            .with_shared_pins(pins.clone());
        if owner.model_size() != expected {
            return Err(native_error());
        }
        Ok((owner, pins, sampling))
    })()
    .map_err(|error| classify(&error, ErrorCode::ComponentLoadFailed))?;
    let (owner, pins, mut sampling) = setup;
    let model = q::StaticModel::new(
        owner,
        rt,
        if role == Component::Dit {
            Some(Kind::BFloat16)
        } else {
            None
        },
    );
    let patcher = model.patcher();
    let mut manager = Manager::new();
    let mut memory = q::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    let result = (|| -> Result<T, ErrorCode> {
        manager
            .load(
                std::slice::from_ref(&patcher),
                budget(role),
                None,
                false,
                false,
                &mut memory,
            )
            .and_then(|_| rt.native.synchronize_device(0))
            .map_err(|error| classify(&error, ErrorCode::ComponentLoadFailed))?;
        let snapshot = model.owner.borrow().execution_parameters();
        let mut exec = Exec {
            native: rt.native.clone(),
            attention: Attention {
                method: AttentionMethod::Pytorch,
                priority_wrapper: true,
                attention_precision: None,
                low_precision_attention: true,
                model_device: Device::Cuda(0),
            },
            step: 0,
            notify,
        };
        forward(&snapshot, &mut sampling, &mut exec)
            .and_then(|value| rt.native.synchronize_device(0).map(|_| value))
            .map_err(|error| classify(&error, ErrorCode::GenerationFailed))
    })();
    let cleanup = manager.unload_all(Device::Cuda(0), &mut memory);
    drop(patcher);
    drop(model);
    drop(manager);
    drop(memory);
    drop(pins);
    drop(sampling);
    let release = rt
        .native
        .synchronize_device(0)
        .and_then(|_| rt.native.cuda_soft_empty_cache(0));
    // Return CPU outputs only after component ownership and registration cleanup.
    let out = result?;
    cleanup.map_err(|error| classify(&error, ErrorCode::GenerationFailed))?;
    release.map_err(|error| classify(&error, ErrorCode::GenerationFailed))?;
    Ok(out)
}
struct Native {
    config: Config,
}
impl Generator for Native {
    fn generate(&mut self, request: &Request, emit: &mut Emit<'_>) -> Result<(), ErrorCode> {
        let start = Instant::now();
        let dev = Device::Cuda(0);
        let phase = |emit: &mut Emit<'_>, phase| {
            emit(
                &Event::Phase {
                    id: request.id.clone(),
                    phase,
                },
                &[],
            )
        };
        phase(emit, Phase::LoadingText)?;
        let models = Models::new(self.config.model_root.clone())?;
        models.verify_all()?;
        // Everything above is admitted-request CPU identity work. The first
        // native/CUDA initialization is deliberately after that boundary.
        let rt = Runtime::frozen_partial(
            unsafe { Kitchen::load(&self.config.kitchen) }
                .map_err(|_| ErrorCode::DeploymentMismatch)?,
        )
        .map_err(|error| classify(&error, ErrorCode::GenerationFailed))?;
        bind_comfy_layouts(&rt);
        rt.backends.policy.disable("triton");
        tch::set_num_threads(2);
        let _grad = tch::no_grad_guard();
        let aimdo = unsafe { Aimdo::load(&self.config.aimdo) }
            .map_err(|_| ErrorCode::DeploymentMismatch)?;
        let mut quiet = |_| Ok(());
        let conditioning = component(
            &rt,
            &aimdo,
            &models,
            Component::Text,
            &mut quiet,
            |params, _, exec| {
                phase(emit, Phase::EncodingText).map_err(|_| native_error())?;
                let mut tokenizer = h3::tokenizer::Tokenizer::frozen()?;
                let tokens = tokenizer.tokenize(&request.prompt, &[], &[], false)?;
                let encoder = h3::text_encoder::TextEncoder::from_parameters(params, true)?;
                unsafe { encoder.encode(&rt, &tokens, dev, Device::Cpu, exec) }
            },
        )?;
        phase(emit, Phase::LoadingDit)?;
        let mut progress = |notice| {
            let event = match notice {
                Notice::Phase(phase) => Event::Phase {
                    id: request.id.clone(),
                    phase,
                },
                Notice::Step(completed) => Event::Progress {
                    id: request.id.clone(),
                    completed,
                    total: p::STEPS,
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                },
            };
            emit(&event, &[]).map_err(|_| native_error())
        };
        let latents = component(
            &rt,
            &aimdo,
            &models,
            Component::Dit,
            &mut progress,
            |params, schedule, exec| {
                (exec.notify)(Notice::Phase(Phase::Sampling))?;
                schedule.sigmas = params["model_sampling.sigmas"].as_tensor()?;
                let sigmas = h3::sampling::simple_sigmas(schedule, 4)?;
                let base = Tensor::f_zeros([1, 32, 12, 16, 24], (Kind::Float, dev))?;
                let latent =
                    h3::streams::NestedTensor::new(h3::preprocessing::empty_streams(&base)?.into());
                let noise = latent.prepare_noise(request.seed, None)?;
                let denoiser =
                    h3::dit::Denoiser::from_parameters(h3::dit::Config::selected()?, params)?;
                let out = unsafe {
                    h3::sampling::sample_euler_cfg1(
                        &rt,
                        &denoiser,
                        schedule,
                        h3::sampling::EulerRequest {
                            latent: &latent,
                            noise: &noise,
                            sigmas: &sigmas,
                            context: &conditioning.context,
                            payload: h3::dit::Payload {
                                text_tags: Some(&conditioning.token_tags),
                                seed: request.seed,
                                ..Default::default()
                            },
                            denoise_mask: None,
                            kind: Kind::BFloat16,
                        },
                        exec,
                    )
                }?;
                let values = out.unbind();
                let values = values.borrow();
                Ok((
                    values[0].as_tensor()?.f_to_device(Device::Cpu)?,
                    values[1].as_tensor()?.f_to_device(Device::Cpu)?,
                ))
            },
        )?;
        drop(conditioning);
        phase(emit, Phase::LoadingVideoVae)?;
        let frames = component(
            &rt,
            &aimdo,
            &models,
            Component::VideoVae,
            &mut quiet,
            |params, _, exec| {
                phase(emit, Phase::DecodingVideo).map_err(|_| native_error())?;
                let config = h3::video_vae::Config {
                    int8_attention_available: rt.int8_attention_available(Some(dev))?,
                    ..Default::default()
                };
                let vae = h3::video_vae::VideoVae::new(params.clone(), config);
                let video = latents.0.f_to_device_(dev, Kind::Half, false, false)?;
                let frames = unsafe { vae.decode(&rt, &video, None, exec) }?;
                if frames.size() != [1, 3, 39, 256, 384] {
                    return Err(native_error());
                }
                Ok(frames.f_to_device(Device::Cpu)?)
            },
        )?;
        for index in 0..p::FRAMES {
            let frame = frames
                .f_get(0)
                .and_then(|v| v.f_select(1, index as i64))
                .and_then(|v| v.f_permute([1, 2, 0]))
                .and_then(|v| v.f_to_kind(Kind::Float))
                .and_then(|v| v.f_contiguous())
                .map_err(|_| ErrorCode::GenerationFailed)?;
            let mut pixels = vec![0f32; p::FRAME_BYTES];
            let n = pixels.len();
            frame
                .f_copy_data(&mut pixels, n)
                .map_err(|_| ErrorCode::GenerationFailed)?;
            let bytes = p::export_rgb8(&pixels).map_err(|_| ErrorCode::GenerationFailed)?;
            emit(
                &Event::VideoFrame {
                    id: request.id.clone(),
                    index,
                },
                &bytes,
            )?;
        }
        drop(frames);
        phase(emit, Phase::LoadingAudioVae)?;
        let pcm = component(
            &rt,
            &aimdo,
            &models,
            Component::AudioVae,
            &mut quiet,
            |params, _, exec| {
                phase(emit, Phase::DecodingAudio).map_err(|_| native_error())?;
                let vae = h3::audio_vae::AudioVae::new(params.clone(), Default::default());
                let audio = latents.1.f_to_device(dev)?;
                let pcm = unsafe { vae.decode(&rt, &audio, exec) }?;
                if pcm.size() != [1, 2, 52000] || pcm.kind() != Kind::Float {
                    return Err(native_error());
                }
                Ok(pcm
                    .f_to_device(Device::Cpu)?
                    .f_transpose(1, 2)?
                    .f_contiguous()?)
            },
        )?;
        let mut samples = vec![0f32; p::AUDIO_SAMPLES as usize * p::CHANNELS as usize];
        let n = samples.len();
        pcm.f_copy_data(&mut samples, n)
            .map_err(|_| ErrorCode::GenerationFailed)?;
        let bytes = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        emit(
            &Event::AudioChunk {
                id: request.id.clone(),
                start: 0,
                samples: p::AUDIO_SAMPLES,
            },
            &bytes,
        )
    }
}
pub fn run() -> Result<(), ErrorCode> {
    let mut output = io::stdout().lock();
    let config = if std::env::args_os().len() != 1 {
        Err(ErrorCode::InvalidRequest)
    } else {
        Config::environment()
    };
    let config = match config {
        Ok(config) => config,
        Err(code) => {
            let _ = wire::write(&mut output, &Event::Error { id: None, code }, &[]);
            return Err(code);
        }
    };
    let deployment = config.deployment.clone();
    wire::serve(
        &mut io::stdin().lock(),
        &mut output,
        deployment,
        move |_| Ok(Native { config }),
    )
}
