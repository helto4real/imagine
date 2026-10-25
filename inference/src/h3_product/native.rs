//! Frozen H3 algorithms with a new private transport adapter; one job per process.
use super::{
    deployment::{self, Config},
    protocol::{self as p, ErrorCode, Event, Phase, Request},
    wire::{self, Emit, Generator},
};
use crate::components as c;
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
use std::{collections::BTreeMap, io, path::PathBuf, time::Instant};
use tch::{Device, Kind, Tensor};
type NResult<T> = Result<T, Error>;
#[cfg(test)]
#[path = "sampler_gpu_test.rs"]
mod sampler_gpu_test;
const PROFILE: &str = include_str!("../../profiles/h3-short-av.json");
fn native_error() -> Error {
    Error::Native("H3 product boundary failed".into())
}
fn classify(error: &Error, fallback: ErrorCode) -> ErrorCode {
    let root = error.root();
    // The message only selects a kind and shape integers; its text is never sent.
    let (kind, sizes) = p::classify_message(&root.to_string());
    let (exception, accelerator_code) = match root {
        Error::ReferenceException { class, .. } => (Some(*class), None),
        Error::AcceleratorException { code, .. } => (None, Some(*code)),
        _ => (None, None),
    };
    let oom = matches!(
        root,
        Error::AcceleratorException { code: 2, .. }
            | Error::ReferenceException {
                class: "OutOfMemoryError",
                ..
            }
    );
    let kind = match root {
        _ if oom => p::FailureKind::OutOfMemory,
        Error::AcceleratorException { .. } if kind == p::FailureKind::Internal => {
            p::FailureKind::Accelerator
        }
        Error::ReferenceException {
            class: "KeyError", ..
        } => p::FailureKind::MissingTensor,
        Error::BindingGap { .. } => p::FailureKind::NotImplemented,
        _ => kind,
    };
    wire::record(kind, exception, accelerator_code, sizes);
    if oom {
        ErrorCode::OutOfMemory
    } else {
        fallback
    }
}
struct Models {
    root: PathBuf,
    profile: Value,
    i2v: bool,
    options: p::options::Options,
}
impl Models {
    fn new(root: PathBuf, i2v: bool, options: p::options::Options) -> Result<Self, ErrorCode> {
        Ok(Self {
            root,
            i2v,
            options,
            profile: serde_json::from_str(PROFILE).map_err(|_| ErrorCode::DeploymentMismatch)?,
        })
    }
    fn path(&self, role: Component) -> Result<PathBuf, ErrorCode> {
        let fallback = if self.i2v && role == Component::Dit {
            "diffusion_models/minimax_h3/minimaxH3.int8.safetensors"
        } else {
            self.profile["checkpoints"][role.name()]
                .as_str()
                .ok_or(ErrorCode::DeploymentMismatch)?
        };
        let components = &self.options.components;
        let selected = match role {
            Component::Dit => components.model.as_deref(),
            Component::Text => components.text_encoder.as_deref(),
            Component::VideoVae => components.video_vae.as_deref(),
            Component::AudioVae => components.audio_vae.as_deref(),
        };
        Ok(p::options::Components::path(&self.root, selected, fallback))
    }
    fn verify(&self, role: Component) -> Result<PathBuf, ErrorCode> {
        let path = self
            .path(role)?
            .canonicalize()
            .map_err(|_| ErrorCode::ComponentLoadFailed)?;
        deployment::checkpoint(&path)?;
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
    Step(p::progress::Report),
    Preview(p::preview::Metadata, Vec<u8>),
    PreviewWarning,
}
struct Exec<'a> {
    native: Kitchen,
    attention: Attention,
    sampler_progress: p::progress::Tracker,
    capture_prediction: bool,
    last_prediction: Option<Tensor>,
    window: Option<super::motion::Window>,
    boundary_window_seen: bool,
    preview: Option<super::preview_sampler::Sampler>,
    notify: &'a mut dyn FnMut(Notice) -> NResult<()>,
}
impl TextEncoderExecution for Exec<'_> {
    fn observe(&mut self, name: &str, value: &Tensor) -> NResult<()> {
        if let Some(index) = name
            .strip_prefix("h3.sampling.")
            .and_then(|s| s.strip_suffix(".denoised"))
        {
            if self.capture_prediction {
                self.last_prediction = Some(value.shallow_clone());
            }
            let index = index.parse::<u32>().map_err(|_| native_error())?;
            let Some(step) = self.sampler_progress.step(index) else {
                return Ok(());
            };
            if let Some(report) = self.sampler_progress.observe(index) {
                (self.notify)(Notice::Step(report))?;
            }
            if let Some(preview) = &mut self.preview {
                match preview.observe(value, step) {
                    Ok(Some((metadata, pixels))) => {
                        (self.notify)(Notice::Preview(metadata, pixels))?
                    }
                    Ok(None) => {}
                    Err(_) if preview.warn_once() => (self.notify)(Notice::PreviewWarning)?,
                    Err(_) => {}
                }
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
        let backend = |q: &Tensor, k: &Tensor, v: &Tensor| unsafe {
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
        };
        if mask.is_none()
            && let Some(window) = &self.window
        {
            self.boundary_window_seen |= window.matches_layout(q, k, v);
            window.run(q, k, v, backend)
        } else {
            backend(q, k, v)
        }
    }
}
impl VideoExecution for Exec<'_> {
    fn free_memory(&mut self, device: Device) -> NResult<u64> {
        let Device::Cuda(i) = device else {
            return c::host_available_memory();
        };
        let [free, _, active, reserved] = self.native.cuda_memory(i)?;
        Ok(free + reserved - active)
    }
}
fn budget(role: Component, shape: p::Shape) -> f64 {
    match role {
        Component::Text => 0.,
        Component::Dit => super::memory_budget::dit(shape).0,
        Component::VideoVae => {
            (9.5 * 30. * shape.height as f64 * shape.width as f64 + 270_000_000.)
                * 2.
                * 1.03
                * (shape.frames as f64 / 39.).max(1.)
        }
        Component::AudioVae => {
            42_000_000. * 4. * 1.03 * (shape.audio_samples() as f64 / 52_000.).max(1.)
        }
    }
}
/// ComfyUI full checkpoints prefix every diffusion-model key; the H3
/// constructors use the bare names.
const DIFFUSION_PREFIX: &str = "model.diffusion_model.";
fn prepare(
    rt: &Runtime,
    role: Component,
    state: &component::ComponentState,
) -> NResult<PreparedParameters> {
    if role == Component::Dit
        && let V::Dict(weights) = state.weights.snapshot()?
        && !weights.is_empty()
        && weights
            .iter()
            .all(|(name, _)| name.starts_with(DIFFUSION_PREFIX))
    {
        let tensors = weights
            .into_iter()
            .map(|(name, value)| {
                Ok((
                    name[DIFFUSION_PREFIX.len()..].to_owned(),
                    value.as_tensor()?,
                ))
            })
            .collect::<NResult<BTreeMap<_, _>>>()?;
        return h3::checkpoint::prepare(rt, role, tensors);
    }
    h3::checkpoint::prepare_normalized(rt, role, state)
}
fn component<T>(
    rt: &Runtime,
    aimdo: &Aimdo,
    models: &Models,
    role: Component,
    shape: p::Shape,
    notify: &mut dyn FnMut(Notice) -> NResult<()>,
    forward: impl FnOnce(&PreparedParameters, &mut ModelSampling, &mut Exec<'_>) -> NResult<T>,
) -> Result<T, ErrorCode> {
    // Recheck readable header structure after admission; tensor constructors validate compatibility.
    let path = models.verify(role)?;
    if role == Component::Text {
        let parameters = (|| -> NResult<_> {
            let state = unsafe {
                component::load_aimdo(
                    rt,
                    aimdo,
                    &path,
                    &Options::Atom(V::None),
                    &StorageAnnotations::default(),
                )
            }?;
            let parameters = prepare(rt, role, &state)?;
            crate::checkpoint::dynamic::prepare_text(
                rt,
                aimdo.path(),
                parameters,
                h3::checkpoint::component_inventory(role)?,
            )
        })()
        .map_err(|error| classify(&error, ErrorCode::ComponentLoadFailed))?;
        let mut sampling =
            h3::sampling::schedule(rt, models.options.shift_video, models.options.shift_audio)
                .map_err(|error| classify(&error, ErrorCode::ComponentLoadFailed))?;
        let mut exec = Exec {
            native: rt.native.clone(),
            attention: Attention {
                method: AttentionMethod::Pytorch,
                priority_wrapper: true,
                attention_precision: None,
                low_precision_attention: true,
                model_device: Device::Cuda(0),
            },
            sampler_progress: p::progress::Tracker::new(p::STEPS),
            capture_prediction: false,
            last_prediction: None,
            window: None,
            preview: None,
            boundary_window_seen: false,
            notify,
        };
        let result = forward(&parameters, &mut sampling, &mut exec);
        // Complete queued uses on both success and failure before destroying
        // the model's virtual address space and its non-owning ATen views.
        let synchronized = rt.native.synchronize_device(0);
        drop(parameters);
        let release = rt.native.cuda_soft_empty_cache(0);
        let out = result.map_err(|error| classify(&error, ErrorCode::GenerationFailed))?;
        synchronized
            .and(release)
            .map_err(|error| classify(&error, ErrorCode::GenerationFailed))?;
        return Ok(out);
    }
    let mut lora_failure = false;
    let setup = (|| -> NResult<_> {
        let annotations = StorageAnnotations::default();
        let state = unsafe {
            component::load_aimdo(rt, aimdo, &path, &Options::Atom(V::None), &annotations)
        }?;
        let parameters = prepare(rt, role, &state)?;
        let sampling =
            h3::sampling::schedule(rt, models.options.shift_video, models.options.shift_audio)?;
        let auxiliary = if role == Component::Dit {
            BTreeMap::from([(
                "model_sampling.sigmas".into(),
                V::tensor(sampling.sigmas.shallow_clone()),
            )])
        } else {
            BTreeMap::new()
        };
        let pins = c::HostPins::new(rt.native.clone())?;
        let patches = if role == Component::Dit {
            Some(
                super::loras::owner(&parameters, &models.root, &models.options.loras)
                    .inspect_err(|_| lora_failure = true)?,
            )
        } else {
            None
        };
        let mut owner = h3::checkpoint::static_component(rt, role, parameters, auxiliary)?;
        if let Some(patches) = patches {
            owner = owner
                .with_patches(patches)
                .inspect_err(|_| lora_failure = true)?;
        }
        let owner = owner.with_shared_pins(pins.clone());
        Ok((owner, pins, sampling))
    })()
    .map_err(|error| {
        classify(
            &error,
            if lora_failure {
                ErrorCode::LoraLoadFailed
            } else {
                ErrorCode::ComponentLoadFailed
            },
        )
    })?;
    let (owner, pins, mut sampling) = setup;
    let model = c::StaticModel::new(
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
    let mut memory = c::StaticMemoryServices::new(rt.native.clone(), pins.clone());
    let result = (|| -> Result<T, ErrorCode> {
        manager
            .load(
                std::slice::from_ref(&patcher),
                budget(role, shape),
                (role == Component::Dit).then(|| super::memory_budget::dit(shape).1),
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
            sampler_progress: p::progress::Tracker::new(p::STEPS),
            capture_prediction: false,
            last_prediction: None,
            window: None,
            preview: None,
            boundary_window_seen: false,
            notify,
        };
        forward(&snapshot, &mut sampling, &mut exec)
            .and_then(|value| rt.native.synchronize_device(0).map(|_| value))
            .map_err(|error| classify(&error, ErrorCode::GenerationFailed))
    })();
    let cleanup = super::lifecycle::release_component(model, patcher, &mut manager);
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
    observer: Option<Observer>,
}
pub type Observer = fn(&str, &Tensor) -> NResult<()>;
impl Generator for Native {
    fn generate(
        &mut self,
        request: &Request,
        emit: &mut Emit<'_>,
        cache_input: &mut dyn io::Read,
    ) -> Result<(), ErrorCode> {
        let observer = self.observer;
        let sampling = request.sampling.clone().unwrap_or_default();
        let output_shape = request.shape;
        let base_shape = sampling.input_shape(output_shape);
        if request.continuation.is_some() {
            wire::stage(p::Stage::PreparingContinuation);
        }
        let (mut shape, head, guide) = request
            .continuation
            .as_ref()
            .map(|c| {
                p::continuation::runtime(base_shape, c.method).ok_or(ErrorCode::InvalidRequest)
            })
            .transpose()?
            .unwrap_or((base_shape, 0, 0));
        let context_tail = request
            .continuation
            .as_ref()
            .map(|c| super::motion::decode(c, &request.continuation_bytes))
            .transpose()
            .map_err(|_| ErrorCode::InvalidRequest)?;
        let context_base = if request.continuation.as_ref().is_some_and(|input| {
            input.method == p::continuation::Method::MotionContextExp
                && input.descriptor.handoff_upscale.is_some()
        }) {
            context_tail
                .as_ref()
                .map(|context| super::motion::area_context(context, base_shape))
                .transpose()
                .map_err(|_| ErrorCode::InvalidRequest)?
        } else {
            None
        };
        let start = Instant::now();
        let dev = Device::Cuda(0);
        let phase = |emit: &mut Emit<'_>, phase: Phase| {
            wire::stage(phase.into());
            emit(
                &Event::Phase {
                    id: request.id.clone(),
                    phase,
                },
                &[],
            )
        };
        phase(emit, Phase::LoadingText)?;
        let models = Models::new(
            self.config.model_root.clone(),
            request.profile == p::Profile::ImageToAvV1,
            request.options.clone(),
        )?;
        models.verify_all()?;
        // Everything above is admitted-request bounded CPU header validation. The first
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
        let schedule_model = h3::sampling::schedule(
            &rt,
            request.options.shift_video,
            request.options.shift_audio,
        )
        .map_err(|_| ErrorCode::InvalidRequest)?;
        let stage_sigmas = super::sampling::resolve(&sampling, |stage| {
            let count = (f64::from(stage.steps) / stage.denoise) as i64;
            let sigmas = crate::sampling::schedules::calculate(
                &schedule_model,
                &stage.scheduler,
                count,
                &rt.native,
                &mut |_| {},
            )?;
            let end = sigmas.size()[0];
            sigmas
                .f_slice(0, (end - i64::from(stage.steps) - 1).max(0), end, 1)
                .map_err(Error::from)
        })
        .map_err(|_| ErrorCode::InvalidRequest)?;
        let second_models = sampling
            .second
            .as_ref()
            .map(|second| {
                let mut options = request.options.stage_two(); // Original model, no internal LoRAs.
                options.components.model = second.model_file.clone().or_else(|| {
                    (second.profile == request.profile)
                        .then(|| request.options.components.model.clone())
                        .flatten()
                });
                Models::new(
                    self.config.model_root.clone(),
                    second.profile == p::Profile::ImageToAvV1,
                    options,
                )
            })
            .transpose()?;
        if let Some(models) = &second_models {
            models.verify(Component::Dit)?;
        }
        let aimdo = unsafe { Aimdo::load(&self.config.aimdo) }
            .map_err(|_| ErrorCode::DeploymentMismatch)?;
        let mut quiet = |_| Ok(());
        if request.references.manifest["boundary_dependency"].is_object()
            && request.references.manifest["frame0_guide"]["enabled"] == true
            && request.image.is_none()
        {
            return Err(ErrorCode::InvalidRequest);
        }
        let start_image = if let Some(image) = &request.image {
            let pillow_path = &self.config.pillow;
            let pillow = unsafe { q::Pillow::open(pillow_path) }
                .map_err(|_| ErrorCode::DeploymentMismatch)?;
            let rgb = pillow
                .resize_rgb(
                    &request.image_rgb,
                    image.width as i32,
                    image.height as i32,
                    3,
                    shape.width as i32,
                    shape.height as i32,
                )
                .map_err(|_| ErrorCode::GenerationFailed)?;
            let values = rgb.iter().map(|v| f32::from(*v) / 255.).collect::<Vec<_>>();
            Some(
                Tensor::f_from_slice(&values)
                    .and_then(|v| v.f_view([1, i64::from(shape.height), i64::from(shape.width), 3]))
                    .map_err(|_| ErrorCode::GenerationFailed)?,
            )
        } else {
            None
        };
        let reference_images =
            super::references::images(&request.references, shape, &self.config.pillow)
                .map_err(|e| classify(&e, ErrorCode::GenerationFailed))?;
        let reference_videos = request
            .references
            .videos
            .iter()
            .map(|v| super::reference_video::prepare(v, shape, &self.config.pillow))
            .collect::<NResult<Vec<_>>>()
            .map_err(|e| classify(&e, ErrorCode::GenerationFailed))?;
        if let Some(observe) = observer {
            for (i, image) in reference_images.iter().enumerate() {
                observe(&format!("reference_image_{i}_prepared"), image)
                    .map_err(|e| classify(&e, ErrorCode::GenerationFailed))?;
            }
        }
        let conditioning = component(
            &rt,
            &aimdo,
            &models,
            Component::Text,
            shape,
            &mut quiet,
            |params, _, exec| {
                phase(emit, Phase::EncodingText).map_err(|_| native_error())?;
                let mut tokenizer = h3::tokenizer::Tokenizer::frozen()?;
                let images = start_image
                    .as_ref()
                    .filter(|_| request.profile == p::Profile::ImageToAvV1)
                    .map(|v| vec![V::tensor(v.shallow_clone())])
                    .unwrap_or_default();
                let reference_values = reference_images
                    .iter()
                    .map(|v| V::tensor(v.shallow_clone()))
                    .collect::<Vec<_>>();
                let mut references = reference_values
                    .iter()
                    .map(h3::tokenizer::Reference::Image)
                    .collect::<Vec<_>>();
                let video_values = reference_videos
                    .iter()
                    .map(super::reference_video::qwen)
                    .collect::<NResult<Vec<_>>>()?
                    .into_iter()
                    .map(V::tensor)
                    .collect::<Vec<_>>();
                let video_times = video_values
                    .iter()
                    .map(|v| {
                        v.as_tensor()
                            .map(|t| (0..t.size()[0]).map(|i| i as f64 / 2.).collect::<Vec<_>>())
                    })
                    .collect::<NResult<Vec<_>>>()?;
                for (i, video) in video_values.iter().enumerate() {
                    if request.references.videos[i].audio.is_some() {
                        references.push(h3::tokenizer::Reference::Audio);
                    }
                    references.push(h3::tokenizer::Reference::Video {
                        frames: video,
                        timestamps: Some(&video_times[i]),
                    });
                }
                references.extend(
                    request
                        .references
                        .audios
                        .iter()
                        .map(|_| h3::tokenizer::Reference::Audio),
                );
                let tokens = tokenizer.tokenize(&request.prompt, &images, &references, false)?;
                let encoder = h3::text_encoder::TextEncoder::from_parameters(params, true)?;
                unsafe { encoder.encode(&rt, &tokens, dev, Device::Cpu, exec) }
            },
        )?;
        let keyframe_latent = if let Some(image) = &start_image {
            phase(emit, Phase::LoadingVideoVae)?;
            wire::stage(p::Stage::EncodingStartImage);
            Some(component(
                &rt,
                &aimdo,
                &models,
                Component::VideoVae,
                shape,
                &mut quiet,
                |params, _, exec| {
                    let vae = h3::video_vae::VideoVae::new(
                        params.clone(),
                        h3::video_vae::Config {
                            int8_attention_available: rt.int8_attention_available(Some(dev))?,
                            ..Default::default()
                        },
                    );
                    let image = super::image::vae_input(image, Kind::Half)?;
                    unsafe { vae.encode(&rt, &image, dev, exec) }?
                        .f_to_device(Device::Cpu)
                        .map_err(Error::from)
                },
            )?)
        } else {
            None
        };
        drop(start_image);
        let mut keys = keyframe_latent
            .as_ref()
            .map(|z| {
                vec![h3::layout::Keyframe {
                    resolved_frame_index: 0,
                    video_t: Some(z.size()[2]),
                    audio_t: None,
                }]
            })
            .unwrap_or_default();
        let mut conditions = keyframe_latent.into_iter().collect::<Vec<_>>();
        let mut audio_conditions = vec![];
        if let (Some(input), Some(context)) = (&request.continuation, &context_tail)
            && input.method == p::continuation::Method::MotionContext
        {
            wire::stage(p::Stage::PreparingMotionContext);
            keys.push(h3::layout::Keyframe {
                resolved_frame_index: i64::from(guide),
                video_t: Some(7),
                audio_t: input.include_audio.then_some(37),
            });
            conditions.push(context.0.shallow_clone());
            if input.include_audio {
                audio_conditions.push(context.1.shallow_clone());
            }
        }
        let mut reference_blocks = vec![];
        if !reference_images.is_empty() {
            phase(emit, Phase::LoadingVideoVae)?;
            wire::stage(p::Stage::EncodingReferences);
            let reference_shape = p::Shape {
                width: reference_images
                    .iter()
                    .map(|v| v.size()[2] as u32)
                    .max()
                    .unwrap(),
                height: reference_images
                    .iter()
                    .map(|v| v.size()[1] as u32)
                    .max()
                    .unwrap(),
                frames: 5,
                ..shape
            };
            let encoded = component(
                &rt,
                &aimdo,
                &models,
                Component::VideoVae,
                reference_shape,
                &mut quiet,
                |params, _, exec| {
                    let vae = h3::video_vae::VideoVae::new(
                        params.clone(),
                        h3::video_vae::Config {
                            int8_attention_available: rt.int8_attention_available(Some(dev))?,
                            ..Default::default()
                        },
                    );
                    reference_images
                        .iter()
                        .map(|image| {
                            let input = super::image::vae_input(image, Kind::Half)?;
                            super::reference_cache::encode(
                                &request.id,
                                super::reference_cache::Encoder {
                                    weights: &models
                                        .verify(Component::VideoVae)
                                        .map_err(|_| native_error())?,
                                    library: &self.config.kitchen,
                                    policy: b"image:fp16:video-vae-v1",
                                },
                                &input,
                                emit,
                                cache_input,
                                || unsafe { vae.encode(&rt, &input, dev, exec) },
                            )
                        })
                        .collect::<NResult<Vec<_>>>()
                },
            )?;
            for (i, z) in encoded.into_iter().enumerate() {
                if let Some(observe) = observer {
                    observe(&format!("reference_image_{i}_latent"), &z)
                        .map_err(|e| classify(&e, ErrorCode::GenerationFailed))?;
                }
                reference_blocks.push(h3::layout::ReferenceBlock {
                    kind: h3::layout::ReferenceKind::Image,
                    latent_t: 0,
                    latent_h: z.size()[3],
                    latent_w: z.size()[4],
                    ref_audio_t: 0,
                });
                conditions.push(z);
            }
        }
        drop(reference_images);
        let video_latents = if reference_videos.is_empty() {
            vec![]
        } else {
            phase(emit, Phase::LoadingVideoVae)?;
            let reference_shape = p::Shape {
                width: reference_videos
                    .iter()
                    .map(|t| t.size()[2] as u32)
                    .max()
                    .unwrap(),
                height: reference_videos
                    .iter()
                    .map(|t| t.size()[1] as u32)
                    .max()
                    .unwrap(),
                frames: reference_videos
                    .iter()
                    .map(|t| t.size()[0] as u32)
                    .max()
                    .unwrap(),
                ..shape
            };
            component(
                &rt,
                &aimdo,
                &models,
                Component::VideoVae,
                reference_shape,
                &mut quiet,
                |params, _, exec| {
                    let vae = h3::video_vae::VideoVae::new(
                        params.clone(),
                        h3::video_vae::Config {
                            int8_attention_available: rt.int8_attention_available(Some(dev))?,
                            ..Default::default()
                        },
                    );
                    reference_videos
                        .iter()
                        .map(|frames| {
                            let input = super::reference_video::vae_input(frames)?;
                            super::reference_cache::encode(
                                &request.id,
                                super::reference_cache::Encoder {
                                    weights: &models
                                        .verify(Component::VideoVae)
                                        .map_err(|_| native_error())?,
                                    library: &self.config.kitchen,
                                    policy: b"video:fp16:video-vae-v1",
                                },
                                &input,
                                emit,
                                cache_input,
                                || unsafe { vae.encode(&rt, &input, dev, exec) },
                            )
                        })
                        .collect::<NResult<Vec<_>>>()
                },
            )?
        };
        drop(reference_videos);
        let paired_count = request
            .references
            .videos
            .iter()
            .filter(|v| v.audio.is_some())
            .count();
        let audio_inputs = request
            .references
            .videos
            .iter()
            .filter_map(|v| v.audio.as_ref())
            .chain(request.references.audios.iter())
            .collect::<Vec<_>>();
        let encoded_audio = if !audio_inputs.is_empty() {
            phase(emit, Phase::LoadingAudioVae)?;
            component(
                &rt,
                &aimdo,
                &models,
                Component::AudioVae,
                shape,
                &mut quiet,
                |params, _, exec| {
                    let vae = h3::audio_vae::AudioVae::new(params.clone(), Default::default());
                    audio_inputs
                        .iter()
                        .enumerate()
                        .map(|(i, input)| {
                            let waveform = super::references::audio(input)?;
                            if let Some(observe) = observer
                                && i >= paired_count
                            {
                                observe(
                                    &format!("reference_audio_{}_resampled", i - paired_count),
                                    &waveform,
                                )?;
                            }
                            super::reference_cache::encode(
                                &request.id,
                                super::reference_cache::Encoder {
                                    weights: &models
                                        .verify(Component::AudioVae)
                                        .map_err(|_| native_error())?,
                                    library: &self.config.kitchen,
                                    policy: b"audio:audio-vae-v1",
                                },
                                &waveform,
                                emit,
                                cache_input,
                                || unsafe { vae.encode(&rt, &waveform.f_to_device(dev)?, exec) },
                            )
                        })
                        .collect::<NResult<Vec<_>>>()
                },
            )?
        } else {
            vec![]
        };
        let mut encoded_audio = encoded_audio.into_iter();
        for (i, z) in video_latents.into_iter().enumerate() {
            let audio = if request.references.videos[i].audio.is_some() {
                Some(encoded_audio.next().ok_or(ErrorCode::GenerationFailed)?)
            } else {
                None
            };
            reference_blocks.push(h3::layout::ReferenceBlock {
                kind: if audio.is_some() {
                    h3::layout::ReferenceKind::VideoAudio
                } else {
                    h3::layout::ReferenceKind::Video
                },
                latent_t: z.size()[2],
                latent_h: z.size()[3],
                latent_w: z.size()[4],
                ref_audio_t: audio.as_ref().map_or(0, |a| a.size()[3]),
            });
            conditions.push(z);
            if let Some(audio) = audio {
                audio_conditions.push(audio);
            }
        }
        for (i, z) in encoded_audio.enumerate() {
            if let Some(observe) = observer {
                observe(&format!("reference_audio_{i}_latent"), &z)
                    .map_err(|e| classify(&e, ErrorCode::GenerationFailed))?;
            }
            reference_blocks.push(h3::layout::ReferenceBlock {
                kind: h3::layout::ReferenceKind::Audio,
                latent_t: 0,
                latent_h: 0,
                latent_w: 0,
                ref_audio_t: z.size()[3],
            });
            audio_conditions.push(z);
        }
        if let Some(observe) = observer {
            let counts = Tensor::f_from_slice(&[
                request.references.images.len() as f32,
                request.references.audios.len() as f32,
                reference_blocks.len() as f32,
                conditions.len() as f32,
                audio_conditions.len() as f32,
            ])
            .map_err(|e| classify(&e.into(), ErrorCode::GenerationFailed))?;
            observe("reference_payload_counts", &counts)
                .map_err(|e| classify(&e, ErrorCode::GenerationFailed))?;
        }
        let mut progress = |notice| {
            if let Notice::Preview(metadata, pixels) = notice {
                return emit(
                    &Event::SamplingPreview {
                        id: request.id.clone(),
                        metadata,
                    },
                    &pixels,
                )
                .map_err(|_| native_error());
            }
            let event = match notice {
                Notice::PreviewWarning => Event::PreviewWarning {
                    id: request.id.clone(),
                    code: "fast_preview_failed".into(),
                },
                Notice::Phase(phase) => {
                    wire::stage(phase.into());
                    Event::Phase {
                        id: request.id.clone(),
                        phase,
                    }
                }
                Notice::Step(report) => Event::Progress {
                    id: request.id.clone(),
                    completed: report.completed,
                    total: report.total,
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                },
                Notice::Preview(..) => unreachable!(),
            };
            emit(&event, &[]).map_err(|_| native_error())
        };
        let mut sampled: Option<(Tensor, Tensor)> = None;
        let mut upscaler: Option<super::upscale::Upscaler> = None;
        let mut sampler_progress = p::progress::Tracker::new(request.total_steps());
        let mut preview_sequence = 0;
        for (index, sigmas) in stage_sigmas.iter().enumerate() {
            if index == 1
                && let Some(upscale) = sampling.second.as_ref().and_then(|s| s.upscale.as_ref())
            {
                progress(Notice::Phase(Phase::LoadingUpscaler))
                    .map_err(|_| ErrorCode::GenerationFailed)?;
                let path = p::options::Components::path(
                    &models.root,
                    upscale.model.as_deref(),
                    "latent_upscale_models/h3_upscaler_lms_v0.1_fp32.safetensors",
                );
                let pair = sampled.as_mut().ok_or(ErrorCode::GenerationFailed)?;
                let operation = (|| -> NResult<()> {
                    if shape.width != output_shape.width || shape.height != output_shape.height {
                        if upscaler.is_none() {
                            upscaler = Some(super::upscale::Upscaler::load(&path)?);
                        }
                        let model = upscaler.as_mut().ok_or_else(native_error)?;
                        progress(Notice::Phase(Phase::UpscalingLatents))?;
                        pair.0 = model.video(&pair.0, upscale.factor, dev)?;
                        rt.native.cuda_soft_empty_cache(0)?;
                        for aligned in conditions.iter_mut().take(keys.len()) {
                            *aligned = model.video(aligned, upscale.factor, dev)?;
                            rt.native.cuda_soft_empty_cache(0)?;
                        }
                    }
                    // Source RandomNoise is seeded again for the resized video only.
                    tch::manual_seed(request.seed as i64);
                    let noise = Tensor::f_randn(pair.0.size(), (Kind::Float, Device::Cpu))?
                        .f_to_kind(pair.0.kind())?;
                    pair.0 = super::sampling::renoise_video(
                        &pair.0.f_to_device(dev)?,
                        &noise.f_to_device(dev)?,
                        stage_sigmas[0].double_value(&[-1]),
                    )?
                    .f_to_device(Device::Cpu)?;
                    Ok(())
                })();
                operation.map_err(|error| classify(&error, ErrorCode::LatentUpscaleFailed))?;
                shape.width = output_shape.width;
                shape.height = output_shape.height;
            }
            let stage = if index == 0 {
                &sampling.first
            } else {
                &sampling
                    .second
                    .as_ref()
                    .ok_or(ErrorCode::InvalidRequest)?
                    .sampling
            };
            let stage_models = if index == 0 {
                &models
            } else {
                second_models.as_ref().ok_or(ErrorCode::InvalidRequest)?
            };
            let renoise = index == 0
                && sampling.second.as_ref().is_some_and(|second| {
                    second.handoff == p::sampling::Handoff::ReNoiseVideoPrediction
                });
            progress(Notice::Phase(Phase::LoadingDit)).map_err(|_| ErrorCode::GenerationFailed)?;
            sampled = Some(component(
                &rt,
                &aimdo,
                stage_models,
                Component::Dit,
                shape,
                &mut progress,
                |params, schedule, exec| {
                    (exec.notify)(Notice::Phase(Phase::Sampling))?;
                    exec.sampler_progress = sampler_progress;
                    let following = stage_sigmas[index + 1..]
                        .iter()
                        .map(|sigmas| sigmas.size()[0] as u32 - 1)
                        .sum();
                    exec.sampler_progress.begin_stage(following);
                    exec.capture_prediction = renoise;
                    schedule.sigmas = params["model_sampling.sigmas"].as_tensor()?;
                    let base = Tensor::f_zeros(
                        [
                            1,
                            32,
                            shape.video_latent_t(),
                            i64::from(shape.height / 16),
                            i64::from(shape.width / 16),
                        ],
                        (Kind::Float, dev),
                    )?;
                    let latent = if let Some((video, audio)) = &sampled {
                        h3::streams::NestedTensor::new(vec![
                            V::tensor(video.f_to_device(dev)?),
                            V::tensor(audio.f_to_device(dev)?),
                        ])
                    } else {
                        h3::streams::NestedTensor::new(
                            h3::preprocessing::empty_streams(&base)?.into(),
                        )
                    };
                    // Stage 1 uses the spatial proxy; Stage 2 restores the exact
                    // original tail after prediction upscaling and video re-noise.
                    let context = if index == 0 {
                        context_base.as_ref().or(context_tail.as_ref())
                    } else {
                        context_tail.as_ref()
                    };
                    let mask =
                        if let (Some(input), Some(context)) = (&request.continuation, context) {
                            wire::stage(p::Stage::PreparingMotionContext);
                            // Reference images sit between the keyframes and the target.
                            let keyframe_rows = keys
                                .iter()
                                .map(|key| {
                                    key.video_t.unwrap_or(0)
                                        * i64::from(shape.width / 32)
                                        * i64::from(shape.height / 32)
                                        + key.audio_t.unwrap_or(0) * 2
                                })
                                .sum();
                            exec.window = Some(super::motion::Window::new(
                                input,
                                shape,
                                conditioning.context.size()[1],
                                keyframe_rows,
                                super::reference_window::rows(&reference_blocks),
                            )?);
                            if input.method == p::continuation::Method::MotionContextExp {
                                let values = latent.unbind();
                                let values = values.borrow();
                                Some(super::motion::prefix(
                                    &values[0].as_tensor()?,
                                    &values[1].as_tensor()?,
                                    context,
                                    input.include_audio,
                                )?)
                            } else {
                                None
                            }
                        } else {
                            if request.references.manifest["boundary_dependency"].is_object()
                                && request.profile != p::Profile::ImageToAvV1
                            {
                                let keyframe_rows = keys
                                    .iter()
                                    .map(|key| {
                                        key.video_t.unwrap_or(0)
                                            * i64::from(shape.width / 32)
                                            * i64::from(shape.height / 32)
                                            + key.audio_t.unwrap_or(0) * 2
                                    })
                                    .sum();
                                exec.window = Some(super::reference_window::prepare(
                                    shape,
                                    conditioning.context.size()[1],
                                    keyframe_rows,
                                    &reference_blocks,
                                )?);
                            }
                            None
                        };
                    wire::stage(p::Stage::Sampling);
                    let noise = if index == 0 {
                        latent.prepare_noise(request.seed, None)?
                    } else {
                        latent.prepare_empty_noise(&rt.native)?
                    };
                    let denoiser =
                        h3::dit::Denoiser::from_parameters(h3::dit::Config::selected()?, params)?;
                    let input = h3::sampling::EulerRequest {
                        latent: &latent,
                        noise: &noise,
                        sigmas,
                        context: &conditioning.context,
                        payload: h3::dit::Payload {
                            text_tags: Some(&conditioning.token_tags),
                            keyframes: &keys,
                            refs: &reference_blocks,
                            video_conditions: &conditions,
                            audio_conditions: &audio_conditions,
                            seed: request.seed,
                            ..Default::default()
                        },
                        denoise_mask: mask.as_ref(),
                        kind: Kind::BFloat16,
                    };
                    let out = unsafe {
                        if let Some(settings) = request.preview.as_ref().filter(|s| s.enabled) {
                            let (_, shapes) = latent.pack()?;
                            exec.preview = Some(super::preview_sampler::Sampler::new(
                                *settings,
                                shapes,
                                f64::from(shape.fps),
                                preview_sequence,
                            )?);
                        }
                        h3::sampling::sample_named_cfg1(
                            &stage.sampler,
                            &rt,
                            &denoiser,
                            schedule,
                            input,
                            exec,
                        )
                    }?;
                    sampler_progress = exec.sampler_progress;
                    if let Some(preview) = &exec.preview {
                        preview_sequence = preview.sequence();
                    }
                    if request.references.boundary_receipt().is_some()
                        && request.profile != p::Profile::ImageToAvV1
                        && !exec.boundary_window_seen
                    {
                        return Err(Error::Native(
                            "Boundary attention window was not applied".into(),
                        ));
                    }
                    let out = if renoise {
                        let (_, shapes) = latent.pack()?;
                        let prediction = exec.last_prediction.take().ok_or_else(native_error)?;
                        let prediction = h3::flow::process_packed_out(
                            &V::tensor(prediction),
                            &base.size(),
                            schedule.audio_scale()?,
                            &rt.native,
                        )?;
                        let prediction = h3::streams::NestedTensor::unpack(&prediction, &shapes)?;
                        let prediction = prediction.unbind().borrow()[0].as_tensor()?;
                        let video = if sampling
                            .second
                            .as_ref()
                            .is_some_and(|s| s.upscale.is_some())
                        {
                            prediction
                        } else {
                            super::sampling::renoise_video(
                                &prediction,
                                &noise.unbind().borrow()[0].as_tensor()?,
                                sigmas.double_value(&[-1]),
                            )?
                        };
                        h3::streams::NestedTensor::new(vec![
                            V::tensor(video),
                            out.unbind().borrow()[1].clone(),
                        ])
                    } else {
                        out
                    };
                    let values = out.unbind();
                    let values = values.borrow();
                    Ok((
                        values[0].as_tensor()?.f_to_device(Device::Cpu)?,
                        values[1].as_tensor()?.f_to_device(Device::Cpu)?,
                    ))
                },
            )?);
            if let Some(report) = sampler_progress.finish_stage() {
                progress(Notice::Step(report)).map_err(|_| ErrorCode::GenerationFailed)?;
            }
        }
        let latents = sampled.ok_or(ErrorCode::GenerationFailed)?;
        if let Some(receipt) = request.references.boundary_receipt() {
            emit(
                &Event::BoundaryApplied {
                    id: request.id.clone(),
                    receipt,
                },
                &[],
            )?;
        }
        if let Some(observe) = observer {
            for (name, value) in [
                ("output_video_latent", &latents.0),
                ("output_audio_latent", &latents.1),
            ] {
                observe(name, value).map_err(|e| classify(&e, ErrorCode::GenerationFailed))?;
            }
        }
        drop(conditioning);
        if request.capture_continuation {
            wire::stage(p::Stage::CapturingContinuation);
            let artifact = super::motion::capture(
                &latents.0,
                &latents.1,
                output_shape,
                shape,
                head,
                p::continuation::HandoffUpscale::from_sampling(&sampling),
            )
            .map_err(|_| ErrorCode::GenerationFailed)?;
            emit(
                &Event::Continuation {
                    id: request.id.clone(),
                    descriptor: artifact.descriptor,
                },
                &artifact.bytes,
            )?;
        }
        phase(emit, Phase::LoadingVideoVae)?;
        let frames = component(
            &rt,
            &aimdo,
            &models,
            Component::VideoVae,
            shape,
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
                if frames.size()
                    != [
                        1,
                        3,
                        i64::from(shape.frames),
                        i64::from(shape.height),
                        i64::from(shape.width),
                    ]
                {
                    return Err(native_error());
                }
                Ok(frames.f_to_device(Device::Cpu)?)
            },
        )?;
        for index in 0..output_shape.frames {
            let frame = frames
                .f_get(0)
                .and_then(|v| v.f_select(1, i64::from(index + head)))
                .and_then(|v| v.f_permute([1, 2, 0]))
                .and_then(|v| v.f_to_kind(Kind::Float))
                .and_then(|v| v.f_contiguous())
                .map_err(|_| ErrorCode::GenerationFailed)?;
            let mut pixels = vec![0f32; shape.frame_bytes()];
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
            shape,
            &mut quiet,
            |params, _, exec| {
                phase(emit, Phase::DecodingAudio).map_err(|_| native_error())?;
                let vae = h3::audio_vae::AudioVae::new(params.clone(), Default::default());
                let audio = latents.1.f_to_device(dev)?;
                let pcm = unsafe { vae.decode(&rt, &audio, exec) }?;
                if pcm.size() != [1, 2, i64::from(shape.native_audio_samples())]
                    || pcm.kind() != Kind::Float
                {
                    return Err(native_error());
                }
                Ok(pcm
                    .f_to_device(Device::Cpu)?
                    .f_transpose(1, 2)?
                    .f_contiguous()?)
            },
        )?;
        let mut samples = vec![0f32; shape.native_audio_samples() as usize * p::CHANNELS as usize];
        let n = samples.len();
        pcm.f_copy_data(&mut samples, n)
            .map_err(|_| ErrorCode::GenerationFailed)?;
        let samples = p::continuation::trim_audio(shape, output_shape, head, samples)
            .map_err(|_| ErrorCode::GenerationFailed)?;
        let bytes = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        for (index, chunk) in bytes.chunks(p::MAX_PAYLOAD / 8 * 8).enumerate() {
            emit(
                &Event::AudioChunk {
                    id: request.id.clone(),
                    start: (index * (p::MAX_PAYLOAD / 8)) as u32,
                    samples: (chunk.len() / 8) as u32,
                },
                chunk,
            )?;
        }
        Ok(())
    }
}
pub fn run() -> Result<(), ErrorCode> {
    run_inner(None)
}
/// Explicit instrumentation entry point used only by the own reference probe example.
pub fn run_observed(observer: Observer) -> Result<(), ErrorCode> {
    run_inner(Some(observer))
}
fn run_inner(observer: Option<Observer>) -> Result<(), ErrorCode> {
    let mut output = io::stdout().lock();
    let config = if std::env::args_os().len() != 1 {
        Err(ErrorCode::InvalidRequest)
    } else {
        Config::environment()
    };
    let config = match config {
        Ok(config) => config,
        Err(code) => {
            let _ = wire::write(
                &mut output,
                &Event::Error {
                    id: None,
                    code,
                    failure: None,
                },
                &[],
            );
            return Err(code);
        }
    };
    let deployment = config.deployment.clone();
    wire::serve(
        &mut io::stdin().lock(),
        &mut output,
        deployment,
        move |_| Ok(Native { config, observer }),
    )
}
