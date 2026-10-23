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
const LOCK: &[u8] = include_bytes!("../../../design/h3-i2v-model-lock.json");
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
    i2v: bool,
}
impl Models {
    fn new(root: PathBuf, i2v: bool) -> Result<Self, ErrorCode> {
        if format!("{:x}", Sha256::digest(LOCK)) != p::MODEL_LOCK_SHA256 {
            return Err(ErrorCode::DeploymentMismatch);
        }
        Ok(Self {
            root,
            i2v,
            lock: serde_json::from_slice(LOCK).map_err(|_| ErrorCode::DeploymentMismatch)?,
            profile: serde_json::from_str(PROFILE).map_err(|_| ErrorCode::DeploymentMismatch)?,
        })
    }
    fn path(&self, role: Component) -> Result<PathBuf, ErrorCode> {
        if self.i2v && role == Component::Dit {
            return Ok(self
                .root
                .join("diffusion_models/minimax_h3/minimaxH3.int8.safetensors"));
        }
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
            .find(|entry| {
                entry["role"]
                    == if self.i2v && role == Component::Dit {
                        "dit_i2v"
                    } else {
                        role.name()
                    }
            })
            .ok_or(ErrorCode::DeploymentMismatch)?;
        if !entry.is_object() {
            return Err(ErrorCode::DeploymentMismatch);
        }
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
    Step(u32),
}
struct Exec<'a> {
    native: Kitchen,
    attention: Attention,
    step: u32,
    step_offset: u32,
    total_steps: u32,
    capture_prediction: bool,
    last_prediction: Option<Tensor>,
    window: Option<super::motion::Window>,
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
            let step = self.step_offset + index.parse::<u32>().map_err(|_| native_error())? + 1;
            if step > self.step {
                if step != self.step + 1 || step > self.total_steps {
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
            window.run(q, k, v, backend)
        } else {
            backend(q, k, v)
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
fn component<T>(
    rt: &Runtime,
    aimdo: &Aimdo,
    models: &Models,
    role: Component,
    shape: p::Shape,
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
    // Recheck readable header structure after admission; tensor constructors validate compatibility.
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
            step: 0,
            step_offset: 0,
            total_steps: p::STEPS,
            capture_prediction: false,
            last_prediction: None,
            window: None,
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
    observer: Option<Observer>,
}
pub type Observer = fn(&str, &Tensor) -> NResult<()>;
impl Generator for Native {
    fn generate(&mut self, request: &Request, emit: &mut Emit<'_>) -> Result<(), ErrorCode> {
        let observer = self.observer;
        let output_shape = request.shape;
        let (shape, head, guide) = request
            .continuation
            .as_ref()
            .map(|c| {
                p::continuation::runtime(output_shape, c.method).ok_or(ErrorCode::InvalidRequest)
            })
            .transpose()?
            .unwrap_or((output_shape, 0, 0));
        let context_tail = request
            .continuation
            .as_ref()
            .map(|c| super::motion::decode(c, &request.continuation_bytes))
            .transpose()
            .map_err(|_| ErrorCode::InvalidRequest)?;
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
        let models = Models::new(
            self.config.model_root.clone(),
            request.profile == p::Profile::ImageToAvV1,
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
        let sampling = request.sampling.clone().unwrap_or_default();
        let schedule_model =
            h3::sampling::selected_schedule(&rt).map_err(|_| ErrorCode::InvalidRequest)?;
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
                Models::new(
                    self.config.model_root.clone(),
                    second.profile == p::Profile::ImageToAvV1,
                )
            })
            .transpose()?;
        if let Some(models) = &second_models {
            models.verify(Component::Dit)?;
        }
        let aimdo = unsafe { Aimdo::load(&self.config.aimdo) }
            .map_err(|_| ErrorCode::DeploymentMismatch)?;
        let mut quiet = |_| Ok(());
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
                            unsafe { vae.encode(&rt, &input, dev, exec) }?
                                .f_to_device(Device::Cpu)
                                .map_err(Error::from)
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
                            unsafe { vae.encode(&rt, &input, dev, exec) }?
                                .f_to_device(Device::Cpu)
                                .map_err(Error::from)
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
                            unsafe { vae.encode(&rt, &waveform.f_to_device(dev)?, exec) }?
                                .f_to_device(Device::Cpu)
                                .map_err(Error::from)
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
            let event = match notice {
                Notice::Phase(phase) => Event::Phase {
                    id: request.id.clone(),
                    phase,
                },
                Notice::Step(completed) => Event::Progress {
                    id: request.id.clone(),
                    completed,
                    total: request.total_steps(),
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                },
            };
            emit(&event, &[]).map_err(|_| native_error())
        };
        let mut sampled: Option<(Tensor, Tensor)> = None;
        let mut step_offset = 0;
        for (index, sigmas) in stage_sigmas.iter().enumerate() {
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
                    exec.step_offset = step_offset;
                    exec.step = step_offset;
                    exec.total_steps = request.total_steps();
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
                    let mask = if let (Some(input), Some(context)) =
                        (&request.continuation, &context_tail)
                    {
                        exec.window = Some(super::motion::Window::new(
                            input,
                            shape,
                            conditioning.context.size()[1],
                        ));
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
                        None
                    };
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
                        if stage.sampler == "euler" {
                            h3::sampling::sample_euler_cfg1(&rt, &denoiser, schedule, input, exec)
                        } else {
                            h3::sampling::sample_res_multistep_cfg1(
                                &rt, &denoiser, schedule, input, exec,
                            )
                        }
                    }?;
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
                        let video = super::sampling::renoise_video(
                            &prediction.unbind().borrow()[0].as_tensor()?,
                            &noise.unbind().borrow()[0].as_tensor()?,
                            sigmas.double_value(&[-1]),
                        )?;
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
            step_offset += sigmas.size()[0] as u32 - 1;
        }
        let latents = sampled.ok_or(ErrorCode::GenerationFailed)?;
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
            let artifact =
                super::motion::capture(&latents.0, &latents.1, output_shape, shape, head)
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
            let _ = wire::write(&mut output, &Event::Error { id: None, code }, &[]);
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
