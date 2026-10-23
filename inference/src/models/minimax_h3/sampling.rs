//! H3's typed two-stream binding to the common AV schedule, parameterization,
//! condition aggregation, KSAMPLER inpaint wrapper and shared solvers.
//! The named CFG=1 entry is the selected probe profile; the common sampler and
//! guidance registries retain their wider independent contracts.
use super::{dit, flow, layout::PackedLayout, streams::NestedTensor};
use crate::{
    context_windows::{self, LatentShapes},
    execution::Runtime,
    execution_value::Value as V,
    models::qwen_image21::{self, TextEncoderExecution},
    native::{self, Error},
    sampling::{
        ksampler,
        model_sampling::{Config, Flavor, ModelSampling},
        parameterization::{Parameterization, Prediction as ParameterizationKind},
        schedules::{self, ScheduleModel},
        solvers,
    },
};
use tch::{Kind, Tensor};

pub fn selected_schedule(rt: &Runtime) -> Result<ModelSampling, Error> {
    ModelSampling::new(
        &rt.native,
        Flavor::AudioVideo,
        Config {
            shift: 12.,
            audio_shift: Some(3.),
            ..Config::default()
        },
        None,
    )
}
pub fn parameterization(schedule: &ModelSampling) -> Parameterization {
    Parameterization {
        prediction: ParameterizationKind::Const,
        sigma_data: schedule.config.sigma_data,
        noise_scale: schedule.config.noise_scale,
    }
}
pub fn token_grid_masks(mask: &Tensor, shapes: &LatentShapes) -> Result<[Tensor; 2], Error> {
    let parts = context_windows::unpack_latents(&V::tensor(mask.shallow_clone()), shapes)?;
    let video = parts[0].as_tensor()?;
    let audio = parts[1].as_tensor()?;
    let s = video.size();
    let rank = s.len();
    let (h, w) = (s[rank - 2], s[rank - 1]);
    let padded = video
        .f_reshape([-1, s[rank - 3], h, w])?
        .f_replication_pad2d([0, (-w).rem_euclid(2), 0, (-h).rem_euclid(2)])?;
    let mut shape = s[..rank - 2].to_vec();
    shape.extend([padded.size()[2] / 2, 2, padded.size()[3] / 2, 2]);
    let video = padded
        .f_reshape(shape.as_slice())?
        .f_amax([-3, -1], false)?
        .f_repeat_interleave_self_int(2, Some(-2), None)?
        .f_repeat_interleave_self_int(2, Some(-1), None)?
        .f_slice(-2, 0, h, 1)?
        .f_slice(-1, 0, w, 1)?;
    let audio = audio
        .f_amax([1], true)?
        .f_expand_as(&audio)?
        .f_contiguous()?;
    Ok([
        video.f_mul_scalar(256.)?.f_ceil()?.f_div_scalar(256.)?,
        audio.f_mul_scalar(256.)?.f_ceil()?.f_div_scalar(256.)?,
    ])
}
pub fn denoise_masks(mask: &Tensor, shapes: &LatentShapes) -> Result<[Option<Tensor>; 2], Error> {
    let [v, a] = token_grid_masks(mask, shapes)?;
    let video = if v.f_amin([], false)?.double_value(&[]) < 1. - 1e-3 {
        Some(native::clone_tensor(
            &v.f_slice(0, 0, 1, 1)?.f_slice(1, 0, 1, 1)?,
        )?)
    } else {
        None
    };
    let audio = if a.f_amin([], false)?.double_value(&[]) < 1. - 1e-3 {
        Some(a.f_slice(0, 0, 1, 1)?.f_amax([1], true)?)
    } else {
        None
    };
    Ok([video, audio])
}
pub fn scale_latent_inpaint(
    schedule: &ModelSampling,
    shapes: &LatentShapes,
    sigma: &Tensor,
    noise: &Tensor,
    latent: &Tensor,
    x: Option<&Tensor>,
    mask: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let mut clean = context_windows::unpack_latents(&V::tensor(latent.shallow_clone()), shapes)?;
    let noises = context_windows::unpack_latents(&V::tensor(noise.shallow_clone()), shapes)?;
    clean[0] = V::tensor(
        clean[0]
            .as_tensor()?
            .f_mul_scalar(0.999)?
            .f_add(&noises[0].as_tensor()?.f_mul_scalar(1. - 0.999)?)?,
    );
    let scale = schedule.audio_scale()?;
    if scale != 1. {
        let sigma_v = sigma.f_clamp_min(1e-6)?;
        let sigma_a = dit::time_shift(
            &sigma_v,
            schedule.config.shift,
            schedule.config.audio_shift.unwrap_or(schedule.config.shift),
        )?;
        let factor = sigma_v.f_div(&sigma_a)?.f_div_scalar(scale)?;
        let audio = clean[1].as_tensor()?;
        let mut shape = factor.size().into_iter().take(1).collect::<Vec<_>>();
        shape.resize(audio.dim(), 1);
        clean[1] =
            V::tensor(audio.f_mul(&factor.f_view(shape.as_slice())?.f_to_kind(audio.kind())?)?);
    }
    let injected = context_windows::pack_latents(&clean)?.0.as_tensor()?;
    let (Some(x), Some(mask)) = (x, mask) else {
        return Ok(injected);
    };
    let grid = token_grid_masks(mask, shapes)?;
    let grid = context_windows::pack_latents(&grid.into_iter().map(V::tensor).collect::<Vec<_>>())?
        .0
        .as_tensor()?;
    let weight = grid
        .f_sub(mask)?
        .f_div(&mask.f_neg()?.f_add_scalar(1.)?.f_clamp_min(1e-6)?)?;
    let weight = weight
        .f_clamp(0., 1.)?
        .f_where_self(&mask.f_lt(1.)?, &weight.f_zeros_like()?)?;
    Ok(injected.f_add(
        &weight
            .f_to_kind(injected.kind())?
            .f_mul(&x.f_sub(&injected)?)?,
    )?)
}

pub struct Conditions {
    pub context: Tensor,
    pub layout: PackedLayout,
    pub masks: [Option<Tensor>; 2],
}
impl Conditions {
    /// # Safety
    /// The model parameters and callbacks satisfy the selected original ABI.
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserve source extra_conds inputs at the typed AV boundary"
    )]
    pub unsafe fn prepare<E: TextEncoderExecution>(
        rt: &Runtime,
        model: &dit::Denoiser,
        context: &Tensor,
        shapes: &LatentShapes,
        payload: dit::Payload<'_>,
        mask: Option<&Tensor>,
        kind: Kind,
        exec: &mut E,
    ) -> Result<Self, Error> {
        let context = unsafe {
            model.preprocess_text(
                rt,
                &context.f_to_device_(context.device(), kind, false, false)?,
                exec,
            )?
        };
        let video = shapes
            .next(0)?
            .ok_or_else(|| Error::Native("Missing H3 video shape".into()))?
            .values()?;
        let audio = shapes
            .next(1)?
            .ok_or_else(|| Error::Native("Missing H3 audio shape".into()))?
            .values()?;
        let layout = PackedLayout::new(
            [
                context.size()[1],
                video[2],
                (video[3] + 1) / 2 * 2,
                (video[4] + 1) / 2 * 2,
                *audio.last().unwrap(),
            ],
            payload.keyframes,
            payload.refs,
        )?;
        let masks = mask
            .map(|m| denoise_masks(m, shapes))
            .transpose()?
            .unwrap_or([None, None]);
        Ok(Self {
            context,
            layout,
            masks,
        })
    }
}
struct Bound<'a, E> {
    rt: &'a Runtime,
    model: &'a dit::Denoiser,
    schedule: &'a ModelSampling,
    shapes: &'a LatentShapes,
    conditions: &'a Conditions,
    payload: dit::Payload<'a>,
    kind: Kind,
    exec: &'a mut E,
}
impl<E: TextEncoderExecution> solvers::Execution for Bound<'_, E> {
    fn denoise(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        cfg: bool,
    ) -> Result<solvers::Prediction, Error> {
        if cfg {
            return Err(Error::BindingGap {
                backend: "h3_cfg1_profile".into(),
                operation: "post-CFG capture requires the common guidance binding".into(),
            });
        }
        self.exec.check_interrupt()?;
        let p = parameterization(self.schedule);
        let input = p.input(sigma, x)?.f_to_kind(self.kind)?;
        let streams = context_windows::unpack_latents(&V::tensor(input), self.shapes)?;
        let t = self.schedule.timestep(sigma)?.f_to_kind(Kind::Float)?;
        let context = self
            .conditions
            .context
            .f_to_device_(x.device(), self.kind, false, false)?;
        let v = self.conditions.masks[0]
            .as_ref()
            .map(|m| m.f_to_device_(x.device(), self.kind, false, false))
            .transpose()?;
        let a = self.conditions.masks[1]
            .as_ref()
            .map(|m| m.f_to_device_(x.device(), self.kind, false, false))
            .transpose()?;
        let payload = dit::Payload {
            layout: Some(&self.conditions.layout),
            ..self.payload
        };
        let output = unsafe {
            self.model.forward(
                self.rt,
                dit::Forward {
                    video: &streams[0].as_tensor()?,
                    audio: &streams[1].as_tensor()?,
                    timestep: &t,
                    context: &context,
                    payload,
                    denoise_mask: v.as_ref(),
                    audio_denoise_mask: a.as_ref(),
                    shifts: [
                        self.schedule.config.shift,
                        self.schedule
                            .config
                            .audio_shift
                            .unwrap_or(self.schedule.config.shift),
                    ],
                },
                self.exec,
            )?
        };
        let output =
            context_windows::pack_latents(&output.into_iter().map(V::tensor).collect::<Vec<_>>())?
                .0
                .as_tensor()?
                .f_to_kind(Kind::Float)?;
        let denoised = p.denoised(sigma, &output, x)?;
        let denoised = qwen_image21::single_full_condition_cfg1(x, &denoised)?;
        Ok(solvers::Prediction {
            denoised,
            uncond_denoised: None,
            cond_denoised: None,
        })
    }
    fn scale_latent_inpaint(
        &mut self,
        _: Parameterization,
        x: &Tensor,
        sigma: &Tensor,
        noise: &Tensor,
        latent: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, Error> {
        scale_latent_inpaint(
            self.schedule,
            self.shapes,
            sigma,
            noise,
            latent,
            Some(x),
            Some(mask),
        )
    }
    fn noise(
        &mut self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: solvers::NoiseKind,
    ) -> Result<Tensor, Error> {
        Err(Error::BindingGap {
            backend: "h3_cfg1_euler_probe".into(),
            operation: "unexpected stochastic noise request".into(),
        })
    }
    fn callback(&mut self, step: qwen_image21::EulerStep<'_>) -> Result<(), Error> {
        self.exec
            .observe(&format!("h3.sampling.{}.x", step.i), step.x)?;
        self.exec
            .observe(&format!("h3.sampling.{}.denoised", step.i), step.denoised)
    }
    fn sample_callback(
        &mut self,
        i: i64,
        denoised: &Tensor,
        x: &Tensor,
        _: i64,
    ) -> Result<(), Error> {
        self.exec.observe(&format!("h3.sampling.{i}.x"), x)?;
        self.exec
            .observe(&format!("h3.sampling.{i}.denoised"), denoised)
    }
}
pub struct EulerRequest<'a> {
    pub latent: &'a NestedTensor,
    pub noise: &'a NestedTensor,
    pub sigmas: &'a Tensor,
    pub context: &'a Tensor,
    pub payload: dit::Payload<'a>,
    pub denoise_mask: Option<&'a Tensor>,
    pub kind: Kind,
}
/// # Safety
/// Selected H3 parameters, two-stream geometry and callbacks satisfy the ABI.
pub unsafe fn sample_euler_cfg1<E: TextEncoderExecution>(
    rt: &Runtime,
    model: &dit::Denoiser,
    schedule: &ModelSampling,
    request: EulerRequest<'_>,
    exec: &mut E,
) -> Result<NestedTensor, Error> {
    unsafe { sample_cfg1(rt, model, schedule, request, exec, false) }
}
/// Director-compatible deterministic res_multistep: eta=0, s_noise=1, CFG++ off.
/// Uses the same AV parameterization and positive-only CFG1 conditions as Euler.
/// # Safety
/// Selected H3 parameters, two-stream geometry and callbacks satisfy the ABI.
pub unsafe fn sample_res_multistep_cfg1<E: TextEncoderExecution>(
    rt: &Runtime,
    model: &dit::Denoiser,
    schedule: &ModelSampling,
    request: EulerRequest<'_>,
    exec: &mut E,
) -> Result<NestedTensor, Error> {
    unsafe { sample_cfg1(rt, model, schedule, request, exec, true) }
}
unsafe fn sample_cfg1<E: TextEncoderExecution>(
    rt: &Runtime,
    model: &dit::Denoiser,
    schedule: &ModelSampling,
    request: EulerRequest<'_>,
    exec: &mut E,
    multistep: bool,
) -> Result<NestedTensor, Error> {
    let (latent, shapes) = request.latent.pack()?;
    let (noise, _) = request.noise.pack()?;
    let scale = schedule.audio_scale()?;
    let video = shapes
        .next(0)?
        .ok_or_else(|| Error::Native("Missing H3 video shape".into()))?
        .values()?;
    let latent = latent.as_tensor()?.f_to_kind(Kind::Float)?;
    // CFGGuider.inner_sample skips the empty latent before latent-format scaling.
    let latent = if latent.f_count_nonzero(None)?.int64_value(&[]) > 0 {
        flow::process_packed_in(&V::tensor(latent), &video, scale, &rt.native)?.as_tensor()?
    } else {
        latent
    };
    let noise = noise
        .as_tensor()?
        .f_to_device_(latent.device(), Kind::Float, false, false)?;
    let payload = dit::Payload {
        audio_scale: scale,
        ..request.payload
    };
    let cond = unsafe {
        Conditions::prepare(
            rt,
            model,
            &request.context.f_to_device(latent.device())?,
            &shapes,
            payload,
            request.denoise_mask,
            request.kind,
            exec,
        )?
    };
    let sigmas = request.sigmas.f_to_device(latent.device())?;
    let mut bound = Bound {
        rt,
        model,
        schedule,
        shapes: &shapes,
        conditions: &cond,
        payload,
        kind: request.kind,
        exec,
    };
    let out = ksampler::sample(
        &mut bound,
        &rt.native,
        ksampler::Inputs {
            sigmas: &sigmas,
            noise: &noise,
            latent: &latent,
            mask: request.denoise_mask,
            random_inpaint: false,
            seed: ksampler::Seed::Integer(i128::from(request.payload.seed)),
            maximum_sigma: schedule.sigma_max()?.double_value(&[]),
            parameterization: parameterization(schedule),
            mask_function: None,
        },
        |e, x, s| {
            if multistep {
                return solvers::res_multistep(e, x, s, parameterization(schedule), 0., 1., false);
            }
            solvers::churn(
                e,
                x,
                s,
                qwen_image21::EulerSettings::default(),
                solvers::ChurnSolver::Euler,
            )
        },
    )?;
    let out = flow::process_packed_out(
        &V::tensor(out.f_to_kind(Kind::Float)?),
        &video,
        scale,
        &rt.native,
    )?;
    NestedTensor::unpack(&out, &shapes)
}
pub fn simple_sigmas(schedule: &ModelSampling, steps: i64) -> Result<Tensor, Error> {
    schedules::simple(schedule, steps)
}
