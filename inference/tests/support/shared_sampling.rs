use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    models::{moss_image::MossSampling, qwen_image21::FluxSampling},
    native::Error,
    sampling::schedules as s,
};
use imagine_inference::{
    models::qwen_image21::{EulerSettings, EulerStep},
    native::Generator,
    sampling::{
        adams, brownian, deis, dpm, exponential, parameterization as p, sde, solvers as solver,
    },
};
use serde_json::Value;
use tch::{Device, Kind, Tensor};

fn data(dev: Device, dtype: Kind, offset: i64) -> Tensor {
    Tensor::arange(12, (Kind::Float, dev))
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .reshape([2, 1, 2, 3])
        .to_kind(dtype)
}
fn parameterization(name: &str) -> p::Prediction {
    match name {
        "EPS" => p::Prediction::Eps,
        "V_PREDICTION" => p::Prediction::V,
        "V_PREDICTION_DDPM" => p::Prediction::VDDPM,
        "EDM" => p::Prediction::Edm,
        "CONST" => p::Prediction::Const,
        "X0" => p::Prediction::X0,
        "IMG_TO_IMG" => p::Prediction::ImageToImage,
        "IMG_TO_IMG_FLOW" => p::Prediction::ImageToImageFlow,
        "IMG_TO_IMG_VELOCITY" => p::Prediction::ImageToImageVelocity,
        "COSMOS_RFLOW" => p::Prediction::CosmosRFlow,
        _ => panic!("Unknown parameterization {name}"),
    }
}
#[test]
fn shared_prediction_parameterizations_match_all_original_algebra_boundaries() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-solvers.json")).unwrap();
    for c in fixture["algebra"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["kind"].as_str().unwrap());
        let shape = c["sigma_shape"].as_array().unwrap();
        let sigma = if shape.is_empty() {
            Tensor::from(0.75f32)
        } else if shape[0] == 1 {
            Tensor::from_slice(&[0.75f32])
        } else {
            Tensor::from_slice(&[0.75f32, 0.25])
        }
        .to_device(dev);
        let model = p::Parameterization {
            prediction: parameterization(c["name"].as_str().unwrap()),
            sigma_data: 0.5,
            noise_scale: 0.875,
        };
        let x = data(dev, dtype, 3);
        let noise = data(dev, dtype, 5);
        let output = data(dev, dtype, 7);
        for (key, y) in [
            ("input", model.input(&sigma, &x)),
            ("denoised", model.denoised(&sigma, &output, &x)),
            ("noise", model.noise_scaling(&sigma, &noise, &x, false)),
            ("noise_max", model.noise_scaling(&sigma, &noise, &x, true)),
            ("inverse", model.inverse_noise_scaling(&sigma, &x)),
        ] {
            check(&V::tensor(y.unwrap()), &c[key], dev, &format!("{c} {key}"));
        }
    }
}
struct Exec<'a> {
    case: &'a Value,
    dev: Device,
    dtype: Kind,
    calls: usize,
    callbacks: usize,
    noises: usize,
    global: Generator,
    independent: Generator,
}
impl Exec<'_> {
    fn predict(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        cfg_pp: bool,
        capture: bool,
    ) -> Result<solver::Prediction, Error> {
        let call = &self.case["calls"][self.calls];
        check(
            &V::tensor(x.shallow_clone()),
            &call["x"],
            self.dev,
            &format!("{} model call {}", self.case["name"], self.calls),
        );
        check(
            &V::tensor(sigma.shallow_clone()),
            &call["sigma"],
            self.dev,
            "model sigma",
        );
        assert_eq!(cfg_pp, call["cfg_pp"].as_bool().unwrap());
        self.calls += 1;
        let mut denoised = x
            .f_mul_scalar(0.125)?
            .f_add(&sigma.f_reshape([2, 1, 1, 1])?.f_mul_scalar(0.25)?)?;
        if self.case["float_output"] == true {
            denoised = denoised.f_to_kind(Kind::Float)?;
        }
        if self.case["float_output"] == "preserve" {
            denoised = denoised.f_to_kind(x.kind())?;
        }
        let uncond_denoised = if capture {
            Some(denoised.f_mul_scalar(0.5)?.f_sub_scalar(0.0625)?)
        } else {
            None
        };
        let cond_denoised = if capture {
            Some(denoised.f_mul_scalar(1.25)?.f_add_scalar(0.03125)?)
        } else {
            None
        };
        Ok(solver::Prediction {
            denoised,
            uncond_denoised,
            cond_denoised,
        })
    }
}
impl solver::Execution for Exec<'_> {
    fn denoise(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        cfg_pp: bool,
    ) -> Result<solver::Prediction, Error> {
        self.predict(x, sigma, cfg_pp, cfg_pp)
    }
    fn denoise_with_capture(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        disable_cfg1: bool,
    ) -> Result<solver::Prediction, Error> {
        self.predict(x, sigma, disable_cfg1, true)
    }
    fn noise(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        next: &Tensor,
        kind: solver::NoiseKind,
    ) -> Result<Tensor, Error> {
        let noise = if kind == solver::NoiseKind::GlobalLike {
            self.global.randn(&x.size(), x.kind())?
        } else {
            self.independent.randn(&[2, 1, 2, 3], self.dtype)?
        };
        let expected = &self.case["noises"][self.noises];
        assert_eq!(
            expected["kind"],
            if kind == solver::NoiseKind::GlobalLike {
                "global_like"
            } else if kind == solver::NoiseKind::Custom {
                "custom"
            } else {
                "independent"
            }
        );
        if kind != solver::NoiseKind::GlobalLike {
            check(
                &V::tensor(sigma.shallow_clone()),
                &expected["sigma"],
                self.dev,
                "noise sigma",
            );
            check(
                &V::tensor(next.shallow_clone()),
                &expected["next"],
                self.dev,
                "noise sigma next",
            );
        }
        check(
            &V::tensor(noise.shallow_clone()),
            &expected["output"],
            self.dev,
            "noise output",
        );
        self.noises += 1;
        Ok(noise)
    }
    fn callback(&mut self, step: EulerStep<'_>) -> Result<(), Error> {
        let expected = &self.case["callbacks"][self.callbacks];
        assert_eq!(step.i, expected["i"].as_i64().unwrap());
        for (key, t) in [
            ("x", step.x),
            ("sigma", step.sigma),
            ("sigma_hat", step.sigma_hat),
            ("denoised", step.denoised),
        ] {
            check(
                &V::tensor(t.shallow_clone()),
                &expected[key],
                self.dev,
                &format!("{} callback {} {key}", self.case["name"], step.i),
            );
        }
        self.callbacks += 1;
        Ok(())
    }
    fn callback_values(&mut self, fields: &[(String, V)]) -> Result<(), Error> {
        let expected = &self.case["callbacks"][self.callbacks];
        for (key, value) in fields {
            if expected[key]["type"] == "tensor" {
                check(
                    value,
                    &expected[key],
                    self.dev,
                    &format!("detailed callback {key}"),
                );
            } else {
                match value {
                    V::Int(n) => assert_eq!(*n, expected[key].as_i64().unwrap(), "callback {key}"),
                    V::Float(n) => assert_eq!(
                        n.to_bits(),
                        expected[key].as_f64().unwrap().to_bits(),
                        "callback {key}"
                    ),
                    _ => panic!("Unexpected callback scalar {key}"),
                }
            }
        }
        assert_eq!(fields.len(), expected.as_object().unwrap().len());
        self.callbacks += 1;
        Ok(())
    }
    fn make_brownian(
        &mut self,
        x: &Tensor,
        min: &Tensor,
        max: &Tensor,
        cpu: bool,
    ) -> Result<brownian::BrownianNoise, Error> {
        brownian::BrownianNoise::new(
            &kitchen(),
            x,
            min,
            max,
            brownian::Seeds::One(901),
            cpu,
            Box::new(|x| Ok(x.shallow_clone())),
        )
    }
    fn observe_noise(
        &mut self,
        noise: &Tensor,
        sigma: &Tensor,
        next: &Tensor,
        kind: solver::NoiseKind,
    ) -> Result<(), Error> {
        let expected = &self.case["noises"][self.noises];
        assert_eq!(
            expected["kind"],
            if kind == solver::NoiseKind::BrownianCpu {
                "brownian_cpu"
            } else {
                "brownian_gpu"
            }
        );
        for (name, tensor) in [("sigma", sigma), ("next", next), ("output", noise)] {
            check(
                &V::tensor(tensor.shallow_clone()),
                &expected[name],
                self.dev,
                "default Brownian sampler noise",
            );
        }
        self.noises += 1;
        Ok(())
    }
    fn percent_to_sigma(&self, percent: f64) -> Result<f64, Error> {
        Ok(FluxSampling {
            shift: 0.69,
            sigmas: Tensor::new(),
        }
        .percent_to_sigma(percent))
    }
}
#[test]
fn original_nested_dpm_wrapper_defaults_preserve_bounds_early_aliases_scalar_errors_predictions_callbacks_and_rng()
 {
    use imagine_inference::sampling::factory::DpmWrapper;
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dpm-wrappers.json")).unwrap();
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let mut exec = Exec {
            case,
            dev,
            dtype,
            calls: 0,
            callbacks: 0,
            noises: 0,
            global: native.generator(dev, 17).unwrap(),
            independent: native
                .generator(dev, if dev == Device::Cpu { 902 } else { 901 })
                .unwrap(),
        };
        let sigmas = match case["mode"].as_str().unwrap() {
            "terminal-zero" => Tensor::from_slice(&[0.9f32, 0.6, 0.3, 0.1, 0.]),
            "terminal-nonzero" => Tensor::from_slice(&[0.9f32, 0.6, 0.3, 0.1, 0.05]),
            "singleton" => Tensor::from_slice(&[0.9f32]),
            "empty" => Tensor::from_slice(&[] as &[f32]),
            "scalar" => Tensor::from(0.9f32),
            "ambiguous" => Tensor::from_slice(&[0.9f32, 0.7, 0.2, 0.]).reshape([2, 2]),
            _ => panic!("own schedule"),
        }
        .to_device(dev);
        let x = data(dev, dtype, 3);
        let wrapper = if case["name"] == "dpm_fast" {
            DpmWrapper::Fast
        } else {
            DpmWrapper::Adaptive
        };
        let result = wrapper.sample_defaults(&mut exec, &native, x.shallow_clone(), &sigmas);
        if case["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("expected wrapper error {case}")),
                case,
            );
        } else {
            let output = result.unwrap_or_else(|error| panic!("{error:?}: {case}"));
            check(
                &V::tensor(output.shallow_clone()),
                &case["output"],
                dev,
                "DPM wrapper output",
            );
            // These controlled normal paths allocate a new result; empty and
            // singleton paths return the original input unchanged.
            assert_eq!(
                output.is_set_to(&x),
                case["returned_input"].as_bool().unwrap()
            );
        }
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "DPM wrapper caller sigmas",
        );
        check(
            &V::tensor(exec.global.state().unwrap()),
            &case["state"],
            dev,
            "DPM wrapper RNG",
        );
        assert_eq!(exec.calls, case["calls"].as_array().unwrap().len());
        assert_eq!(exec.callbacks, case["callbacks"].as_array().unwrap().len());
        assert_eq!(exec.noises, case["noises"].as_array().unwrap().len());
    }
}

#[test]
fn original_advanced_adaptive_dpm_preserves_controller_tolerances_custom_noise_info_early_returns_predictions_callbacks_and_rng()
 {
    use imagine_inference::sampling::dpm_wrapper::{self, AdaptiveResult};
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dpm-advanced.json")).unwrap();
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let mut exec = Exec {
            case,
            dev,
            dtype,
            calls: 0,
            callbacks: 0,
            noises: 0,
            global: native.generator(dev, 17).unwrap(),
            independent: native
                .generator(dev, if dev == Device::Cpu { 902 } else { 901 })
                .unwrap(),
        };
        let values: Vec<f32> = case["sigmas"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let sigmas = Tensor::from_slice(&values).to_device(dev);
        let x = data(dev, dtype, 3);
        let options = &case["options"];
        let defaults = dpm::Adaptive::default();
        let settings = dpm::Adaptive {
            order: options["order"].as_i64().unwrap_or(defaults.order),
            rtol: options["rtol"].as_f64().unwrap_or(defaults.rtol),
            atol: options["atol"].as_f64().unwrap_or(defaults.atol),
            initial_step: options["h_init"].as_f64().unwrap_or(defaults.initial_step),
            p: options["pcoeff"].as_f64().unwrap_or(defaults.p),
            i: options["icoeff"].as_f64().unwrap_or(defaults.i),
            d: options["dcoeff"].as_f64().unwrap_or(defaults.d),
            accept_safety: options["accept_safety"]
                .as_f64()
                .unwrap_or(defaults.accept_safety),
            stochastic: dpm::Stochastic {
                eta: options["eta"].as_f64().unwrap_or(0.),
                noise: options["s_noise"].as_f64().unwrap_or(1.),
            },
        };
        let result = dpm_wrapper::adaptive(
            &mut exec,
            &native,
            x.shallow_clone(),
            &sigmas,
            settings,
            options["return_info"] == true,
        );
        if case["error"].is_string() {
            reference_error(
                result.err().unwrap_or_else(|| panic!("expected {case}")),
                case,
            );
        } else {
            let (output, info) = match result.unwrap_or_else(|error| panic!("{error:?}: {case}")) {
                AdaptiveResult::Samples(value) => (value, Value::Null),
                AdaptiveResult::WithInfo(value, info) => (
                    value,
                    serde_json::json!({"steps":info.steps,"nfe":info.nfe,"n_accept":info.accepted,"n_reject":info.rejected}),
                ),
            };
            check(
                &V::tensor(output.shallow_clone()),
                &case["output"],
                dev,
                "advanced adaptive output",
            );
            assert_eq!(info, case["info"], "info {case}");
            assert_eq!(
                output.is_set_to(&x),
                case["returned_input"].as_bool().unwrap()
            );
        }
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "advanced adaptive caller sigma",
        );
        check(
            &V::tensor(exec.global.state().unwrap()),
            &case["state"],
            dev,
            "advanced adaptive RNG",
        );
        assert_eq!(exec.calls, case["calls"].as_array().unwrap().len());
        assert_eq!(exec.callbacks, case["callbacks"].as_array().unwrap().len());
        assert_eq!(exec.noises, case["noises"].as_array().unwrap().len());
    }
}

#[test]
fn original_advanced_sampler_providers_preserve_cfg_flags_stage_geometry_sde_overrides_tau_scaler_calls_failures_and_rng()
 {
    use std::cell::Cell;
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-options.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 240);
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let options = &case["options"];
        let mode = case["mode"].as_str().unwrap();
        let name = case["name"].as_str().unwrap();
        let mut exec = Exec {
            case,
            dev,
            dtype,
            calls: 0,
            callbacks: 0,
            noises: 0,
            global: native.generator(dev, 17).unwrap(),
            independent: native
                .generator(dev, if dev == Device::Cpu { 902 } else { 901 })
                .unwrap(),
        };
        let x = data(dev, dtype, 3);
        let sigmas =
            Tensor::from_slice(&[0.9f32, 0.6, 0.3, 0.1, case["end"].as_f64().unwrap() as f32])
                .to_device(dev);
        let model = p::Parameterization {
            prediction: if case["flow"] == true {
                p::Prediction::Const
            } else {
                p::Prediction::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let provider_count = Cell::new(0usize);
        let input = |value: &Tensor| -> Result<&Value, Error> {
            let index = provider_count.get();
            provider_count.set(index + 1);
            let expected = &case["providers"][index];
            check(
                &V::tensor(value.shallow_clone()),
                &expected["input"],
                dev,
                "custom provider input",
            );
            if (mode == "scaler-failure" && index == 2) || (mode == "tau-failure" && index == 1) {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "own scalar provider failure".into(),
                });
            }
            Ok(expected)
        };
        let scaler = |value: &Tensor| -> Result<Tensor, Error> {
            let expected = input(value)?;
            let output = value.f_pow_tensor_scalar(2)?.f_add_scalar(1.)?;
            check(
                &V::tensor(output.shallow_clone()),
                &expected["output"],
                dev,
                "custom scaler output",
            );
            Ok(output)
        };
        let tau = |value: &Tensor| -> Result<f64, Error> {
            let expected = input(value)?;
            let output = if mode == "tau-zero" { 0f64 } else { 0.5 };
            assert_eq!(
                output.to_bits(),
                expected["output"].as_f64().unwrap().to_bits()
            );
            Ok(output)
        };
        let result = match name {
            "gradient_estimation" | "gradient_estimation_cfg_pp" => solver::gradient_estimation(
                &mut exec,
                x,
                &sigmas,
                options["ge_gamma"].as_f64().unwrap(),
                name.ends_with("cfg_pp") || options["cfg_pp"] == true,
            ),
            "seeds_3" => exponential::seeds_3(
                &mut exec,
                x,
                &sigmas,
                model,
                exponential::Seeds3 {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: options["s_noise"].as_f64().unwrap(),
                    r1: options["r_1"].as_f64().unwrap(),
                    r2: options["r_2"].as_f64().unwrap(),
                },
            ),
            "er_sde" => exponential::er_sde(
                &mut exec,
                x,
                &sigmas,
                model,
                options["s_noise"].as_f64().unwrap(),
                options["max_stage"].as_i64().unwrap(),
                Some(&scaler),
            ),
            "sa_solver" | "sa_solver_pece" => adams::sample(
                &mut exec,
                x,
                &sigmas,
                model,
                adams::Settings {
                    noise: options["s_noise"].as_f64().unwrap(),
                    predictor: options["predictor_order"].as_i64().unwrap(),
                    corrector: options["corrector_order"].as_i64().unwrap(),
                    pece: name.ends_with("pece"),
                    simple_order_2: options["simple_order_2"] == true,
                },
                Some(&tau),
            ),
            "dpmpp_sde" => sde::dpmpp_sde(
                &mut exec,
                x,
                &sigmas,
                model,
                sde::Settings {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: options["s_noise"].as_f64().unwrap(),
                    source: sde::Source::Custom,
                },
                options["r"].as_f64().unwrap(),
            ),
            "dpmpp_2m_sde" | "dpmpp_2m_sde_heun" => sde::dpmpp_2m_sde(
                &mut exec,
                x,
                &sigmas,
                model,
                sde::Settings {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: options["s_noise"].as_f64().unwrap(),
                    source: sde::Source::Custom,
                },
                options["solver_type"].as_str().unwrap(),
            ),
            _ => panic!("own profile"),
        };
        if case["error"].is_string() {
            reference_error(
                result.err().unwrap_or_else(|| panic!("expected {case}")),
                case,
            );
        } else {
            check(
                &V::tensor(result.unwrap_or_else(|error| panic!("{error:?}: {case}"))),
                &case["output"],
                dev,
                "advanced sampler output",
            );
        }
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "advanced sampler caller sigmas",
        );
        check(
            &V::tensor(exec.global.state().unwrap()),
            &case["state"],
            dev,
            "advanced sampler global RNG",
        );
        assert_eq!(exec.calls, case["calls"].as_array().unwrap().len());
        assert_eq!(exec.callbacks, case["callbacks"].as_array().unwrap().len());
        assert_eq!(exec.noises, case["noises"].as_array().unwrap().len());
        assert_eq!(
            provider_count.get(),
            case["providers"].as_array().unwrap().len()
        );
    }
}

#[test]
fn shared_solvers_preserve_original_predictions_callbacks_noise_and_rng_draws() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-solvers.json")).unwrap();
    for c in fixture["solvers"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["kind"].as_str().unwrap());
        let mut exec = Exec {
            case: c,
            dev,
            dtype,
            calls: 0,
            callbacks: 0,
            noises: 0,
            global: native.generator(dev, 17).unwrap(),
            independent: native
                .generator(dev, if dev == Device::Cpu { 902 } else { 901 })
                .unwrap(),
        };
        let sigmas = Tensor::from_slice(&[
            c["first"].as_f64().unwrap_or(0.9) as f32,
            0.6,
            0.3,
            0.1,
            c["end"].as_f64().unwrap() as f32,
        ])
        .to_device(dev);
        let x = data(dev, dtype, 3);
        let name = c["name"].as_str().unwrap();
        let options = &c["options"];
        let model = p::Parameterization {
            prediction: if c["flow"] == true {
                p::Prediction::Const
            } else {
                p::Prediction::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let y = match name {
            "euler" | "heun" | "heunpp2" | "dpm_2" => solver::churn(
                &mut exec,
                x,
                &sigmas,
                EulerSettings {
                    churn: options["s_churn"].as_f64().unwrap(),
                    tmin: options["s_tmin"].as_f64().unwrap_or(0.),
                    tmax: options["s_tmax"].as_f64().unwrap_or(f64::INFINITY),
                    noise: options["s_noise"].as_f64().unwrap_or(1.),
                },
                match name {
                    "euler" => solver::ChurnSolver::Euler,
                    "heun" => solver::ChurnSolver::Heun,
                    "heunpp2" => solver::ChurnSolver::HeunPP2,
                    _ => solver::ChurnSolver::Dpm2,
                },
            ),
            "euler_ancestral" | "dpm_2_ancestral" => solver::sample_ancestral(
                &mut exec,
                x,
                &sigmas,
                model,
                options["eta"].as_f64().unwrap(),
                options["s_noise"].as_f64().unwrap(),
                if name == "euler_ancestral" {
                    solver::AncestralSolver::Euler
                } else {
                    solver::AncestralSolver::Dpm2
                },
            ),
            "dpmpp_2m" | "dpmpp_2m_cfg_pp" => {
                solver::dpmpp_2m(&mut exec, x, &sigmas, name.ends_with("cfg_pp"))
            }
            "gradient_estimation" | "gradient_estimation_cfg_pp" => {
                solver::gradient_estimation(&mut exec, x, &sigmas, 2., name.ends_with("cfg_pp"))
            }
            "euler_ancestral_cfg_pp" | "euler_cfg_pp" => solver::euler_cfg_pp(
                &mut exec,
                x,
                &sigmas,
                model,
                if name == "euler_cfg_pp" {
                    0.
                } else {
                    options["eta"].as_f64().unwrap()
                },
                if name == "euler_cfg_pp" {
                    0.
                } else {
                    options["s_noise"].as_f64().unwrap()
                },
            ),
            "dpmpp_2s_ancestral" | "dpmpp_2s_ancestral_cfg_pp" => solver::dpmpp_2s_ancestral(
                &mut exec,
                x,
                &sigmas,
                model,
                options["eta"].as_f64().unwrap(),
                options["s_noise"].as_f64().unwrap(),
                name.ends_with("cfg_pp"),
            ),
            "res_multistep"
            | "res_multistep_cfg_pp"
            | "res_multistep_ancestral"
            | "res_multistep_ancestral_cfg_pp" => solver::res_multistep(
                &mut exec,
                x,
                &sigmas,
                model,
                if name.contains("ancestral") {
                    options["eta"].as_f64().unwrap()
                } else {
                    0.
                },
                options["s_noise"].as_f64().unwrap_or(1.),
                name.ends_with("cfg_pp"),
            ),
            "ddpm" => solver::ddpm(&mut exec, x, &sigmas),
            "lcm" => solver::lcm(
                &mut exec,
                x,
                &sigmas,
                model,
                solver::LcmSettings {
                    noise: options["s_noise"].as_f64().unwrap_or(1.),
                    noise_end: options["s_noise_end"].as_f64(),
                    noise_clip_std: options["noise_clip_std"].as_f64().unwrap_or(0.),
                },
            ),
            "ipndm" => solver::ipndm(&mut exec, x, &sigmas, 4),
            "ipndm_v" => solver::ipndm_v(&mut exec, x, &sigmas, 4),
            "cfgpp_ud10_ab" => solver::cfgpp_ud10_ab(&mut exec, x, &sigmas, model, &mut |_| {}),
            "deis" => deis::sample(&mut exec, &native, x, &sigmas, 3, "tab"),
            "uni_pc" => {
                imagine_inference::sampling::unipc::sample(&mut exec, &native, x, &sigmas, "bh1")
            }
            "uni_pc_bh2" => {
                imagine_inference::sampling::unipc::sample_bh2(&mut exec, &native, x, &sigmas)
            }
            "sa_solver" | "sa_solver_pece" => adams::sample(
                &mut exec,
                x,
                &sigmas,
                model,
                adams::Settings {
                    noise: options["s_noise"].as_f64().unwrap_or(1.),
                    predictor: options["predictor_order"].as_i64().unwrap_or(3),
                    corrector: options["corrector_order"].as_i64().unwrap_or(4),
                    pece: name.ends_with("pece"),
                    simple_order_2: options["simple_order_2"].as_bool().unwrap_or(false),
                },
                None,
            ),
            "dpm_fast" | "dpm_adaptive" => {
                let (min, max, n) = dpm::bounds(&sigmas).unwrap().unwrap();
                let settings = dpm::Stochastic {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: options["s_noise"].as_f64().unwrap(),
                };
                if name == "dpm_fast" {
                    dpm::fast(&mut exec, &native, x, &min, &max, n, settings)
                } else {
                    dpm::adaptive(
                        &mut exec,
                        &native,
                        x,
                        &min,
                        &max,
                        dpm::Adaptive {
                            stochastic: settings,
                            ..Default::default()
                        },
                    )
                    .map(|(x, _)| x)
                }
            }
            "lms" => solver::lms(&mut exec, x, &sigmas, 4, &mut |_| {}),
            "seeds_2" => exponential::seeds_2(
                &mut exec,
                x,
                &sigmas,
                model,
                exponential::Seeds2 {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: options["s_noise"].as_f64().unwrap(),
                    r: options["r"].as_f64().unwrap(),
                },
                options["solver_type"].as_str().unwrap(),
            ),
            "seeds_3" => exponential::seeds_3(
                &mut exec,
                x,
                &sigmas,
                model,
                exponential::Seeds3 {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: options["s_noise"].as_f64().unwrap(),
                    ..Default::default()
                },
            ),
            "exp_heun_2_x0" | "exp_heun_2_x0_sde" => exponential::seeds_2(
                &mut exec,
                x,
                &sigmas,
                model,
                exponential::Seeds2 {
                    eta: if name == "exp_heun_2_x0" {
                        0.
                    } else {
                        options["eta"].as_f64().unwrap()
                    },
                    noise: if name == "exp_heun_2_x0" {
                        0.
                    } else {
                        options["s_noise"].as_f64().unwrap()
                    },
                    r: 1.,
                },
                "phi_2",
            ),
            "er_sde" => exponential::er_sde(
                &mut exec,
                x,
                &sigmas,
                model,
                options["s_noise"].as_f64().unwrap(),
                options["max_stage"].as_i64().unwrap(),
                None,
            ),
            name if name.starts_with("dpmpp_") && name.contains("_sde") => {
                let settings = sde::Settings {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: options["s_noise"].as_f64().unwrap(),
                    source: if options["_custom"] == true {
                        if name.ends_with("_gpu") {
                            sde::Source::CustomGpu
                        } else {
                            sde::Source::Custom
                        }
                    } else if name.ends_with("_gpu") {
                        sde::Source::Gpu
                    } else {
                        sde::Source::Cpu
                    },
                };
                if name.starts_with("dpmpp_2m") {
                    sde::dpmpp_2m_sde(
                        &mut exec,
                        x,
                        &sigmas,
                        model,
                        settings,
                        if name.contains("heun") {
                            "heun"
                        } else {
                            "midpoint"
                        },
                    )
                } else if name.starts_with("dpmpp_3m") {
                    sde::dpmpp_3m_sde(&mut exec, x, &sigmas, model, settings)
                } else {
                    sde::dpmpp_sde(&mut exec, x, &sigmas, model, settings, 0.5)
                }
            }
            _ => panic!("Unbound solver {name}"),
        };
        if c["error"].is_string() {
            reference_error(
                y.err()
                    .unwrap_or_else(|| panic!("Expected original solver error {c}")),
                c,
            );
        } else {
            check(
                &V::tensor(y.unwrap_or_else(|e| panic!("{e}: {c}"))),
                &c["output"],
                dev,
                &format!("Final output {c}"),
            );
        }
        check(
            &V::tensor(exec.global.state().unwrap()),
            &c["state"],
            dev,
            "Global RNG draw order",
        );
        if !c["post_sigmas"].is_null() {
            check(
                &V::tensor(sigmas),
                &c["post_sigmas"],
                dev,
                "Caller sigma mutation",
            );
        }
        assert_eq!(exec.calls, c["calls"].as_array().unwrap().len());
        assert_eq!(exec.callbacks, c["callbacks"].as_array().unwrap().len());
        assert_eq!(exec.noises, c["noises"].as_array().unwrap().len());
    }
}

#[test]
fn shared_schedules_original_registry_primitives_dtypes_and_errors_match() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-schedules.json")).unwrap();
    let previous = native.default_dtype().unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let dtype = kind(c["kind"].as_str().unwrap());
        let dev = device(c["device"].as_str().unwrap());
        let n = c["n"].as_i64().unwrap();
        let name = c["name"].as_str().unwrap();
        let flux = FluxSampling::qwen21().unwrap();
        let flow = MossSampling::moss().unwrap();
        native.set_default_dtype(dtype).unwrap();
        let mut warnings = Vec::new();
        let actual = if c["primitive"] == true {
            let options = (dtype, dev);
            match name {
                "karras" => s::karras(n, 0.0125, 12.5, 3., options),
                "exponential" => s::exponential(n, 0.0125, 12.5, options),
                "polyexponential" => s::polyexponential(n, 0.0125, 12.5, 1.5, options),
                "vp" => s::vp(n, 19.9, 0.1, 1e-3, options),
                "laplace" => s::laplace(n, 0.0125, 12.5, 0.25, 0.5, options),
                _ => panic!("Unbound primitive {name}"),
            }
        } else {
            let model: &dyn s::ScheduleModel = if c["model"] == "flux" { &flux } else { &flow };
            s::calculate(model, name, n, &native, &mut |m| {
                warnings.push(m.to_owned())
            })
        };
        native.set_default_dtype(previous).unwrap();
        let note = format!("{c}");
        if c["error"].is_object() {
            match actual.expect_err(&note) {
                Error::ReferenceException { class, message } => {
                    assert_eq!(class, c["error"]["class_name"].as_str().unwrap(), "{note}");
                    assert_eq!(message, c["error"]["message"].as_str().unwrap(), "{note}");
                }
                Error::Torch(e) => {
                    assert_eq!(c["error"]["class_name"], "RuntimeError", "{note}");
                    assert!(
                        e.to_string()
                            .contains(c["error"]["message"].as_str().unwrap()),
                        "{e}: {note}"
                    );
                }
                other => panic!("Unexpected {other}: {note}"),
            }
        } else {
            check(
                &V::tensor(actual.unwrap_or_else(|e| panic!("{e}: {note}"))),
                &c["output"],
                dev,
                &note,
            );
        }
    }
    assert_eq!(s::NAMES.len(), 9);
    let mut messages = Vec::new();
    let result = s::calculate(
        &FluxSampling::qwen21().unwrap(),
        "not-a-schedule",
        7,
        &native,
        &mut |m| messages.push(m.to_owned()),
    );
    assert!(matches!(
        result,
        Err(Error::ReferenceException {
            class: "ValueError",
            ..
        })
    ));
    assert_eq!(messages, ["error invalid scheduler not-a-schedule"]);
    assert_eq!(native.default_dtype().unwrap(), previous);
}

#[test]
fn unipc_advanced_prediction_threshold_vary_coeff_orders_and_errors_match() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-unipc-advanced.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["kind"].as_str().unwrap());
        let mut exec = Exec {
            case: c,
            dev,
            dtype,
            calls: 0,
            callbacks: 0,
            noises: 0,
            global: native.generator(dev, 17).unwrap(),
            independent: native.generator(dev, 901).unwrap(),
        };
        let x = data(dev, dtype, 3);
        let x = if c["thresholding"] == true {
            x.f_mul_scalar(8).unwrap()
        } else {
            x
        };
        let times = Tensor::from_slice(&[0.9f32, 0.6, 0.3, 0.1, 0.05]).to_device(dev);
        let mut pc = imagine_inference::sampling::unipc::UniPc {
            execution: &mut exec,
            native: &native,
            schedule: &imagine_inference::sampling::unipc::SigmaConvert,
            settings: imagine_inference::sampling::unipc::Settings {
                predict_x0: c["predict_x0"].as_bool().unwrap(),
                thresholding: c["thresholding"].as_bool().unwrap(),
                max_val: 1.,
                variant: c["variant"].as_str().unwrap().into(),
            },
        };
        let out = pc.multistep(x, &times, c["order"].as_i64().unwrap(), true);
        if c["error"].is_string() {
            reference_error(
                out.err()
                    .unwrap_or_else(|| panic!("Expected source error {c}")),
                c,
            );
        } else {
            check(
                &V::tensor(out.unwrap_or_else(|e| panic!("{e}: {c}"))),
                &c["output"],
                dev,
                &format!("Advanced UniPC {c}"),
            );
        }
        assert_eq!(exec.calls, c["calls"].as_array().unwrap().len());
        assert_eq!(exec.callbacks, c["callbacks"].as_array().unwrap().len());
    }
}
