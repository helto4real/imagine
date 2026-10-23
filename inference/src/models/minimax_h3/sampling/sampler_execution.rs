//! Per-call sampler noise ownership. Kernels and KSAMPLER remain shared.
use crate::{
    execution_value::Value,
    models::qwen_image21::EulerStep,
    native::{Error, Kitchen, normal_from_source},
    sampling::{
        brownian::{BrownianNoise, Seeds},
        default_call,
        locals::SourceLocals,
        noise::DefaultNoise,
        parameterization::Parameterization,
        solvers::{Execution, NoiseKind, Prediction},
    },
};
use tch::Tensor;

pub(super) fn random_inpaint(name: &str) -> bool {
    name == "ddim"
}
fn kernel(name: &str) -> &str {
    if random_inpaint(name) { "euler" } else { name }
}

/// Single full-area positive condition plus an absent negative condition.
/// Matches conditioning_batch::calculate's output/count normalization followed
/// by guidance::cfg. The non-capturing path retains its original helper exactly.
pub(super) fn positive_prediction(
    x: &Tensor,
    model_denoised: &Tensor,
    capture: bool,
) -> Result<Prediction, Error> {
    use crate::models::qwen_image21;
    if !capture {
        return Ok(Prediction {
            denoised: qwen_image21::single_full_condition_cfg1(x, model_denoised)?,
            uncond_denoised: None,
            cond_denoised: None,
        });
    }
    let mult = x.f_ones_like()?.f_mul_scalar(1.)?;
    let mut cond = x.f_zeros_like()?;
    let mut count = x.f_ones_like()?.f_mul_scalar(1e-37)?;
    let _ = cond.f_add_(&model_denoised.f_mul(&mult)?)?;
    let _ = count.f_add_(&mult)?;
    let _ = cond.f_div_(&count)?;
    let mut uncond = x.f_zeros_like()?;
    let _ = uncond.f_div_(&x.f_ones_like()?.f_mul_scalar(1e-37)?)?;
    Ok(Prediction {
        denoised: qwen_image21::cfg_combine(&cond, &uncond, 1.)?,
        uncond_denoised: Some(uncond),
        cond_denoised: Some(cond),
    })
}

/// The default noise closure captures the initial solver tensor. Its independent
/// generator is created only for kernels which request that closure; global-like
/// churn noise and Brownian owners keep their own original RNG contracts.
struct Seeded<'a> {
    inner: &'a mut dyn Execution,
    native: &'a Kitchen,
    source: Tensor,
    seed: u64,
    independent: Option<DefaultNoise>,
}
impl Execution for Seeded<'_> {
    fn set_denoise_mask(&mut self, mask: Option<&Tensor>) -> Result<(), Error> {
        self.inner.set_denoise_mask(mask)
    }
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg: bool) -> Result<Prediction, Error> {
        self.inner.denoise(x, sigma, cfg)
    }
    fn denoise_with_capture(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        disable: bool,
    ) -> Result<Prediction, Error> {
        self.inner.denoise_with_capture(x, sigma, disable)
    }
    fn noise(
        &mut self,
        x: &Tensor,
        _: &Tensor,
        _: &Tensor,
        kind: NoiseKind,
    ) -> Result<Tensor, Error> {
        match kind {
            NoiseKind::GlobalLike => normal_from_source(x, None),
            NoiseKind::Independent => {
                if self.independent.is_none() {
                    self.independent = Some(DefaultNoise::new(
                        self.native,
                        &self.source,
                        Some(i128::from(self.seed)),
                    )?);
                }
                self.independent.as_mut().unwrap().sample()
            }
            _ => Err(Error::BindingGap {
                backend: "h3_sampler_noise".into(),
                operation: format!("No caller-provided {kind:?} noise sampler"),
            }),
        }
    }
    fn make_brownian(
        &mut self,
        x: &Tensor,
        min: &Tensor,
        max: &Tensor,
        cpu: bool,
    ) -> Result<BrownianNoise, Error> {
        BrownianNoise::new(
            self.native,
            x,
            min,
            max,
            Seeds::One(self.seed),
            cpu,
            Box::new(|time| Ok(time.shallow_clone())),
        )
    }
    fn callback(&mut self, step: EulerStep<'_>) -> Result<(), Error> {
        self.inner.callback(step)
    }
    fn callback_values(&mut self, fields: &[(String, Value)]) -> Result<(), Error> {
        self.inner.callback_values(fields)
    }
    fn sample_callback(
        &mut self,
        i: i64,
        denoised: &Tensor,
        x: &Tensor,
        total: i64,
    ) -> Result<(), Error> {
        self.inner.sample_callback(i, denoised, x, total)
    }
    fn eps_callback(&mut self) -> Result<(), Error> {
        self.inner.eps_callback()
    }
    fn observe_noise(
        &mut self,
        noise: &Tensor,
        sigma: &Tensor,
        next: &Tensor,
        kind: NoiseKind,
    ) -> Result<(), Error> {
        self.inner.observe_noise(noise, sigma, next, kind)
    }
    fn percent_to_sigma(&self, percent: f64) -> Result<f64, Error> {
        self.inner.percent_to_sigma(percent)
    }
    fn warning(&mut self, message: &str) {
        self.inner.warning(message)
    }
    fn source_locals(&mut self, frame: &SourceLocals, event: &str) -> Result<(), Error> {
        self.inner.source_locals(frame, event)
    }
    fn scale_latent_inpaint(
        &mut self,
        p: Parameterization,
        x: &Tensor,
        sigma: &Tensor,
        noise: &Tensor,
        latent: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, Error> {
        self.inner
            .scale_latent_inpaint(p, x, sigma, noise, latent, mask)
    }
}

pub(super) fn run(
    name: &str,
    inner: &mut dyn Execution,
    native: &Kitchen,
    x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    seed: u64,
) -> Result<Tensor, Error> {
    let mut exec = Seeded {
        inner,
        native,
        source: x.shallow_clone(),
        seed,
        independent: None,
    };
    let mut warnings = vec![];
    let result = default_call::sample(
        kernel(name),
        &mut exec,
        native,
        x,
        sigmas,
        model,
        &mut |message| warnings.push(message.to_owned()),
    );
    for message in warnings {
        exec.warning(&message);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{ksampler, parameterization::Prediction as Mode};
    use serde_json::Value as Json;
    use sha2::{Digest, Sha256};
    use std::path::Path;
    use tch::{Device, Kind};

    struct Predictor {
        callbacks: Vec<i64>,
        native: Kitchen,
    }
    impl Execution for Predictor {
        fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg: bool) -> Result<Prediction, Error> {
            let denoised = x
                .f_mul_scalar(0.125)?
                .f_add(&sigma.f_reshape([2, 1, 1, 1])?.f_mul_scalar(0.25)?)?;
            if cfg {
                return positive_prediction(x, &denoised, true);
            }
            Ok(Prediction {
                denoised,
                uncond_denoised: None,
                cond_denoised: None,
            })
        }
        fn denoise_with_capture(
            &mut self,
            x: &Tensor,
            sigma: &Tensor,
            _: bool,
        ) -> Result<Prediction, Error> {
            self.denoise(x, sigma, true)
        }
        fn noise(
            &mut self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: NoiseKind,
        ) -> Result<Tensor, Error> {
            panic!("Noise must be supplied by the owned adapter")
        }
        fn callback(&mut self, step: EulerStep<'_>) -> Result<(), Error> {
            self.callbacks.push(step.i);
            Ok(())
        }
        fn callback_values(&mut self, fields: &[(String, Value)]) -> Result<(), Error> {
            // UniPC emits the short i/x/denoised schema. The real product uses
            // KSAMPLER::Inpaint's identical index projection before Bound.
            let Some((_, Value::Int(index))) = fields.iter().find(|(key, _)| key == "i") else {
                return Err(Error::Native("Synthetic callback missing index".into()));
            };
            self.callbacks.push(*index);
            Ok(())
        }
        fn sample_callback(&mut self, i: i64, _: &Tensor, _: &Tensor, _: i64) -> Result<(), Error> {
            self.callbacks.push(i);
            Ok(())
        }
        fn percent_to_sigma(&self, percent: f64) -> Result<f64, Error> {
            use crate::sampling::model_sampling::{Flavor, ModelSampling};
            let schedule = ModelSampling {
                flavor: Flavor::AudioVideo,
                config: super::super::schedule_config(12., 3.),
                sigmas: Tensor::new(),
                log_sigmas: None,
                num_timesteps: 0,
                cosine_s: None,
                initial_alpha: None,
                skip_steps: 1,
            };
            match schedule.percent_to_sigma(&self.native, percent)? {
                Value::Float(value) => Ok(value),
                _ => unreachable!("AV percent_to_sigma returns a scalar"),
            }
        }
    }
    fn native() -> Kitchen {
        unsafe {
            Kitchen::load(Path::new(
                &std::env::var("IMAGINE_KITCHEN_LIBRARY")
                    .expect("Explicit own CPU bridge required"),
            ))
            .unwrap()
        }
    }
    fn data(offset: i64) -> Tensor {
        Tensor::arange(12, (Kind::Float, Device::Cpu))
            .f_add_scalar(offset)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
    }
    fn check(tensor: &Tensor, expected: &Json) {
        assert_eq!(
            tensor.size(),
            serde_json::from_value::<Vec<i64>>(expected["shape"].clone()).unwrap()
        );
        let flat = tensor.contiguous().view([-1]).view_dtype(Kind::Uint8);
        let mut bytes = vec![0_u8; flat.numel()];
        let count = flat.numel();
        flat.copy_data_u8(&mut bytes, count);
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            expected["payload_sha256"].as_str().unwrap()
        );
    }
    fn mode() -> Parameterization {
        Parameterization {
            prediction: Mode::Const,
            sigma_data: 1.,
            noise_scale: 0.875,
        }
    }

    #[test]
    fn ddim_is_the_only_random_inpaint_alias_and_unknown_names_are_not_replaced() {
        for name in crate::sampling::plan::SAMPLERS {
            assert_eq!(random_inpaint(name), name == "ddim");
            assert_eq!(kernel(name), if name == "ddim" { "euler" } else { name });
        }
        assert_eq!(kernel("own-unknown-solver"), "own-unknown-solver");
    }

    #[test]
    #[ignore = "Requires explicitly selected own CPU Kitchen library; no CUDA/model use"]
    fn euler_multistep_and_heun_match_existing_original_const_oracles() {
        let native = native();
        let fixture: Json =
            serde_json::from_str(include_str!("../../../../fixtures/shared-solvers.json")).unwrap();
        for name in ["euler", "res_multistep", "heun"] {
            let case = fixture["solvers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| {
                    c["name"] == name
                        && c["device"] == "cpu"
                        && c["kind"] == "torch.float32"
                        && c["flow"] == true
                        && c["float_output"] == false
                        && c["first"] == 0.9
                        && c["end"] == 0.
                        && (!c["options"]["s_churn"].is_number() || c["options"]["s_churn"] == 0.)
                })
                .unwrap();
            let mut exec = Predictor {
                callbacks: vec![],
                native: native.clone(),
            };
            let sigmas = Tensor::from_slice(&[0.9_f32, 0.6, 0.3, 0.1, 0.]);
            let actual = run(name, &mut exec, &native, data(3), &sigmas, mode(), 901).unwrap();
            check(&actual, &case["output"]);
            assert_eq!(exec.callbacks, vec![0, 1, 2, 3]);
        }
        let mut exec = Predictor {
            callbacks: vec![],
            native: native.clone(),
        };
        let sigma = Tensor::from_slice(&[0.9_f32, 0.6, 0.]);
        assert!(
            run(
                "euler_cfg_pp",
                &mut exec,
                &native,
                data(3),
                &sigma,
                mode(),
                901
            )
            .is_ok()
        );
        assert!(matches!(
            run(
                "own-unknown-solver",
                &mut exec,
                &native,
                data(3),
                &sigma,
                mode(),
                901
            ),
            Err(Error::BindingGap { .. })
        ));
    }

    #[test]
    #[ignore = "Requires explicitly selected own CPU Kitchen library; no CUDA/model use"]
    fn independent_and_brownian_noise_match_saved_original_cpu_payloads() {
        let native = native();
        let source = data(3);
        let mut inner = Predictor {
            callbacks: vec![],
            native: native.clone(),
        };
        let mut exec = Seeded {
            inner: &mut inner,
            native: &native,
            source: source.shallow_clone(),
            seed: 901,
            independent: None,
        };
        let fixture: Json = serde_json::from_str(include_str!(
            "../../../../fixtures/shared-noise-quadrature.json"
        ))
        .unwrap();
        let case = fixture["noises"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["device"] == "cpu" && c["kind"] == "torch.float32" && c["seed"] == "901")
            .unwrap();
        let before = crate::native::global_rng_state(Device::Cpu).unwrap();
        for expected in case["outputs"].as_array().unwrap() {
            let actual = exec
                .noise(
                    &Tensor::zeros([1], (Kind::Float, Device::Cpu)),
                    &Tensor::from(0.9_f32),
                    &Tensor::from(0.5_f32),
                    NoiseKind::Independent,
                )
                .unwrap();
            check(&actual, expected);
        }
        assert!(before.equal(&crate::native::global_rng_state(Device::Cpu).unwrap()));
        let fixture: Json =
            serde_json::from_str(include_str!("../../../../fixtures/shared-brownian.json"))
                .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["device"] == "cpu"
                    && c["kind"] == "torch.float32"
                    && c["cpu"] == true
                    && c["transform"] == "identity"
                    && c["seed"] == 901
            })
            .unwrap();
        let mut tree = exec
            .make_brownian(
                &source,
                &Tensor::from(0.05_f32),
                &Tensor::from(0.95_f32),
                true,
            )
            .unwrap();
        for query in case["queries"].as_array().unwrap().iter().take(3) {
            let noise = tree
                .sample(
                    &Tensor::from(query["start"].as_f64().unwrap() as f32),
                    &Tensor::from(query["end"].as_f64().unwrap() as f32),
                    &mut |_| {},
                )
                .unwrap();
            check(&noise, &query["output"]);
        }
        tch::manual_seed(17);
        let expected = Tensor::randn([3, 2], (Kind::Float, Device::Cpu));
        tch::manual_seed(17);
        let actual = exec
            .noise(
                &Tensor::zeros([3, 2], (Kind::Float, Device::Cpu)),
                &Tensor::from(1.),
                &Tensor::from(0.),
                NoiseKind::GlobalLike,
            )
            .unwrap();
        assert!(actual.equal(&expected));
    }

    #[test]
    #[ignore = "Requires explicitly selected own CPU Kitchen library; no CUDA/model use"]
    fn ddim_and_euler_preserve_existing_ksampler_mask_and_rng_oracles() {
        let native = native();
        let old = native.default_dtype().unwrap();
        native.set_default_dtype(Kind::Float).unwrap();
        let fixture: Json =
            serde_json::from_str(include_str!("../../../../fixtures/shared-ksampler.json"))
                .unwrap();
        for name in ["euler", "ddim"] {
            let case = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| {
                    c["name"] == name
                        && c["device"] == "cpu"
                        && c["kind"] == "torch.float32"
                        && c["flow"] == true
                        && c["mask_mode"] == "fixed"
                })
                .unwrap();
            let sigmas = Tensor::from_slice(&[0.9_f32, 0.6, 0.3, 0.1, 0.]);
            let noise = data(3);
            let latent = data(5);
            let mask = data(7).abs();
            let mut exec = Predictor {
                callbacks: vec![],
                native: native.clone(),
            };
            tch::manual_seed(17);
            let actual = ksampler::sample(
                &mut exec,
                &native,
                ksampler::Inputs {
                    sigmas: &sigmas,
                    noise: &noise,
                    latent: &latent,
                    mask: Some(&mask),
                    random_inpaint: random_inpaint(name),
                    seed: ksampler::Seed::Integer(901),
                    maximum_sigma: 1.,
                    parameterization: mode(),
                    mask_function: None,
                },
                |e, x, s| run(name, e, &native, x, s, mode(), 901),
            )
            .unwrap();
            check(&actual, &case["output"]);
            check(
                &crate::native::global_rng_state(Device::Cpu).unwrap(),
                &case["cpu_rng"],
            );
            assert_eq!(exec.callbacks, vec![0, 1, 2, 3]);
        }
        native.set_default_dtype(old).unwrap();
    }

    #[test]
    #[ignore = "Requires explicitly selected own CPU Kitchen library; no CUDA/model use"]
    fn registered_defaults_dispatch_without_fallback_with_real_empty_condition_capture() {
        let native = native();
        let sigmas = Tensor::from_slice(&[0.9_f32, 0.6, 0.3, 0.1, 0.]);
        let mut successful = 0;
        for name in crate::sampling::plan::SAMPLERS {
            let mut exec = Predictor {
                callbacks: vec![],
                native: native.clone(),
            };
            tch::manual_seed(17);
            let output = run(name, &mut exec, &native, data(3), &sigmas, mode(), 901);
            let output = output.unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(output.size(), [2, 1, 2, 3], "{name}");
            assert_eq!(output.isfinite().all().int64_value(&[]), 1, "{name}");
            assert!(exec.callbacks.iter().all(|index| *index >= 0), "{name}");
            assert!(
                exec.callbacks.windows(2).all(|pair| pair[1] > pair[0]),
                "{name}: {:?}",
                exec.callbacks
            );
            eprintln!(
                "own CPU dispatch {name}: callback_indices={:?}",
                exec.callbacks
            );
            successful += 1;
        }
        assert_eq!(successful, 45);
    }

    #[test]
    fn positive_capture_matches_actual_cond_batch_none_and_common_guidance_bit_for_bit() {
        use crate::{
            conditioning::AreaCondition, conditioning_batch as batch, patching::options::Options,
            sampling::guidance,
        };
        struct Model {
            prediction: Tensor,
            calls: usize,
        }
        impl batch::Model for Model {
            fn prepare_hook_keyframe(
                &mut self,
                _: &Tensor,
                _: &batch::Hooks,
                _: &Options,
            ) -> Result<(), Error> {
                panic!("No hooks authored")
            }
            fn prepare_state(&mut self, _: &Tensor, _: &Options) -> Result<(), Error> {
                Ok(())
            }
            fn free_memory(&mut self, _: Device) -> Result<f64, Error> {
                Ok(f64::INFINITY)
            }
            fn memory_required(
                &mut self,
                _: &[i64],
                _: &[(String, Vec<Value>)],
            ) -> Result<f64, Error> {
                Ok(0.)
            }
            fn apply_hooks(&mut self, _: Option<&batch::Hooks>) -> Result<Options, Error> {
                Ok(Options::dictionary(vec![]))
            }
            fn apply_model(&mut self, call: batch::Call<'_>, _: &Options) -> Result<Tensor, Error> {
                assert_eq!(
                    call.indices,
                    [0],
                    "The absent negative branch must not call the model"
                );
                self.calls += 1;
                Ok(self.prediction.shallow_clone())
            }
            fn warning(&mut self, message: &str) {
                panic!("Unexpected warning {message}")
            }
        }
        for kind in [Kind::Float, Kind::Double] {
            let input = data(3).to_kind(kind);
            let prediction = Tensor::from_slice(&[
                -0_f32, 0., 1e-38, -1e-38, 0.125, -0.125, 1., -1., 0.25, -0.25, 3.5, -3.5,
            ])
            .view([2, 1, 2, 3])
            .to_kind(kind);
            let mut model = Model {
                prediction: prediction.shallow_clone(),
                calls: 0,
            };
            let records = [batch::Record {
                area: AreaCondition {
                    area: None,
                    strength: 1.,
                    mask: None,
                    mask_strength: 1.,
                    timestep_start: None,
                    timestep_end: None,
                },
                conditions: vec![],
                hooks: None,
                control: None,
                gligen: None,
                uuid: Value::Text("own-positive".into()),
                default: false,
            }];
            let time = Tensor::from_slice(&[0.75_f32, 0.75]);
            let options = Options::dictionary(vec![]);
            let original =
                batch::calculate(&mut model, &[Some(&records), None], &input, &time, &options)
                    .unwrap();
            assert_eq!(model.calls, 1);
            let captured = positive_prediction(&input, &prediction, true).unwrap();
            let same_bits = |a: &Tensor, b: &Tensor| {
                assert!(
                    a.contiguous()
                        .view_dtype(Kind::Uint8)
                        .equal(&b.contiguous().view_dtype(Kind::Uint8))
                )
            };
            same_bits(captured.cond_denoised.as_ref().unwrap(), &original[0]);
            same_bits(captured.uncond_denoised.as_ref().unwrap(), &original[1]);
            let expected = guidance::cfg(
                &mut guidance::NoHooks,
                guidance::Context {
                    x: &input,
                    timestep: &time,
                    scale: 1.,
                    options: &options,
                    cond: Some(&()),
                    uncond: None,
                },
                &original[0],
                &original[1],
            )
            .unwrap();
            same_bits(&captured.denoised, &expected);
            let regular = positive_prediction(&input, &prediction, false).unwrap();
            same_bits(&regular.denoised, &expected);
            assert!(regular.cond_denoised.is_none() && regular.uncond_denoised.is_none());
        }
    }
}
