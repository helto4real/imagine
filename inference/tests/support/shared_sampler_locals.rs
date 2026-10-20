use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{
    execution_value::Value as V,
    models::qwen_image21::{EulerSettings, EulerStep},
    native::{Error, Generator},
    sampling::{
        locals::SourceLocals,
        parameterization::{Parameterization, Prediction as ParameterizationKind},
        solvers::{self as s, Execution, NoiseKind, Prediction},
    },
};
use serde_json::Value;
use tch::{Kind, Tensor};
struct Exec<'a> {
    case: &'a Value,
    event: usize,
    rng: Generator,
    initial_kind: Kind,
}
impl Execution for Exec<'_> {
    fn percent_to_sigma(&self, percent: f64) -> Result<f64, Error> {
        Ok(imagine_inference::models::qwen_image21::FluxSampling {
            shift: 0.69,
            sigmas: Tensor::new(),
        }
        .percent_to_sigma(percent))
    }
    fn source_locals(&mut self, locals: &SourceLocals, event: &str) -> Result<(), Error> {
        let expected = &self.case["events"][self.event];
        assert_eq!(expected["event"], event);
        let expected = expected["locals"].as_object().unwrap();
        let mut groups = Vec::new();
        let mut count = 0;
        for (name, tensor) in locals.tensors() {
            let v = expected.get(name).unwrap_or_else(|| {
                panic!("Unexpected source local {name} for {}", self.case["name"])
            });
            check(
                &V::tensor(tensor.shallow_clone()),
                &v["value"],
                tensor.device(),
                &format!("{} {} {event} local {name}", self.case["name"], self.event),
            );
            let info = imagine_inference::native::storage_info(tensor)?;
            let index = if let Some(index) = groups.iter().position(|p| *p == info.0) {
                index
            } else {
                groups.push(info.0);
                groups.len() - 1
            };
            assert_eq!(
                index as u64,
                v["storage_group"].as_u64().unwrap(),
                "storage alias {name}"
            );
            assert_eq!(
                info.1 as u64,
                v["storage_bytes"].as_u64().unwrap(),
                "storage allocation {name}"
            );
            count += 1;
        }
        assert_eq!(
            count,
            expected.len(),
            "Missing source local at event {}: {:?}",
            self.event,
            expected.keys()
        );
        self.event += 1;
        Ok(())
    }
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg: bool) -> Result<Prediction, Error> {
        let y = x
            .f_mul_scalar(0.125)?
            .f_add(&sigma.f_reshape([2, 1, 1, 1])?.f_mul_scalar(0.25)?)?;
        let uncond = cfg
            .then(|| y.f_mul_scalar(0.5)?.f_sub_scalar(0.0625))
            .transpose()?;
        Ok(Prediction {
            denoised: y,
            uncond_denoised: uncond,
            cond_denoised: None,
        })
    }
    fn noise(
        &mut self,
        x: &Tensor,
        _s: &Tensor,
        _t: &Tensor,
        kind: NoiseKind,
    ) -> Result<Tensor, Error> {
        if matches!(kind, NoiseKind::Independent | NoiseKind::Custom) {
            self.rng.randn(&[2, 1, 2, 3], self.initial_kind)
        } else {
            assert_eq!(kind, NoiseKind::GlobalLike);
            self.rng.randn(&x.size(), x.kind())
        }
    }
    fn callback(&mut self, _step: EulerStep<'_>) -> Result<(), Error> {
        Ok(())
    }
    fn denoise_with_capture(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        _disable_cfg1: bool,
    ) -> Result<Prediction, Error> {
        let mut prediction = self.denoise(x, sigma, false)?;
        if self.case["cond_only"] == true {
            prediction.cond_denoised = Some(
                prediction
                    .denoised
                    .f_mul_scalar(1.25)?
                    .f_add_scalar(0.03125)?,
            );
        } else {
            prediction.uncond_denoised = Some(
                prediction
                    .denoised
                    .f_mul_scalar(0.5)?
                    .f_sub_scalar(0.0625)?,
            );
        }
        Ok(prediction)
    }
}
#[test]
fn original_dpmpp_2m_and_two_stage_sde_inner_frames_preserve_scalar_replacements_heun_midpoint_gpu_custom_noise_errors_and_rng()
 {
    use imagine_inference::sampling::sde::{self, Source};
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals9.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 776);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let name = case["name"].as_str().unwrap();
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas = Tensor::from_slice(&[
            case["first"].as_f64().unwrap() as f32,
            0.6,
            0.3,
            0.1,
            case["end"].as_f64().unwrap() as f32,
        ])
        .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let model = Parameterization {
            prediction: if case["flow"] == true {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let settings = sde::Settings {
            eta: case["eta"].as_f64().unwrap(),
            noise: case["noise"].as_f64().unwrap(),
            source: if name.ends_with("_gpu") {
                Source::CustomGpu
            } else {
                Source::Custom
            },
        };
        let result = if name.starts_with("dpmpp_2m") {
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
        } else {
            sde::dpmpp_sde(
                &mut exec,
                x,
                &sigmas,
                model,
                settings,
                case["r"].as_f64().unwrap(),
            )
        };
        if case["error"].is_string() {
            super::binding_tests::reference_error(
                result.err().unwrap_or_else(|| panic!("expected {case}")),
                case,
            );
        } else {
            check(
                &V::tensor(result.unwrap()),
                &case["output"],
                dev,
                "2M/two-stage SDE frame output",
            );
        }
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "2M/two-stage SDE caller sigmas",
        );
        check(
            &V::tensor(exec.rng.state().unwrap()),
            &case["state"],
            dev,
            "2M/two-stage SDE RNG",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}

#[test]
fn original_dpmpp_3m_sde_inner_frames_retain_bounds_prediction_and_step_history_gpu_custom_noise_and_exact_rng()
 {
    use imagine_inference::sampling::sde::{self, Source};
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals8.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 256);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas = Tensor::from_slice(&[
            case["first"].as_f64().unwrap() as f32,
            0.6,
            0.3,
            0.1,
            case["end"].as_f64().unwrap() as f32,
        ])
        .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let model = Parameterization {
            prediction: if case["flow"] == true {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let output = sde::dpmpp_3m_sde(
            &mut exec,
            x,
            &sigmas,
            model,
            sde::Settings {
                eta: case["eta"].as_f64().unwrap(),
                noise: case["noise"].as_f64().unwrap(),
                source: if case["name"] == "dpmpp_3m_sde_gpu" {
                    Source::CustomGpu
                } else {
                    Source::Custom
                },
            },
        )
        .unwrap();
        check(
            &V::tensor(output),
            &case["output"],
            dev,
            "3M SDE frame output",
        );
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "3M SDE caller sigmas",
        );
        check(
            &V::tensor(exec.rng.state().unwrap()),
            &case["state"],
            dev,
            "3M SDE RNG",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}

#[test]
fn original_cfgpp_ud10_and_history_options_preserve_conditional_fallback_numpy_aliases_derivative_lists_and_zero_extrapolation_buffers()
 {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals7.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 192);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let options = &case["options"];
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas = Tensor::from_slice(&[
            case["first"].as_f64().unwrap() as f32,
            0.6,
            0.3,
            0.1,
            case["end"].as_f64().unwrap() as f32,
        ])
        .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let model = Parameterization {
            prediction: if case["flow"] == true {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let output = s::cfgpp_history(
            &mut exec,
            x,
            &sigmas,
            model,
            s::CfgHistory {
                history_weight: options["history_weight"].as_f64().unwrap(),
                zero_weight: options["zero_weight"].as_f64(),
                zero_order: options["zero_order"].as_i64().unwrap(),
                uncond_history_weight: options["uncond_history_weight"].as_f64().unwrap(),
            },
            &mut |message| panic!("unexpected fixture warning {message}"),
        )
        .unwrap();
        check(
            &V::tensor(output),
            &case["output"],
            dev,
            "CFG++ history frame output",
        );
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "CFG++ history original caller sigma",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}

#[test]
fn original_deis_retains_coefficient_list_scalar_aliases_history_buffers_orders_modes_and_source_error_prefixes()
 {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals6.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 80);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas =
            Tensor::from_slice(&[0.9f32, 0.6, 0.3, 0.1, case["end"].as_f64().unwrap() as f32])
                .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let result = imagine_inference::sampling::deis::sample(
            &mut exec,
            &native,
            x.shallow_clone(),
            &sigmas,
            case["order"].as_i64().unwrap(),
            case["mode"].as_str().unwrap(),
        );
        if case["error"].is_string() {
            super::binding_tests::reference_error(
                result.err().unwrap_or_else(|| panic!("expected {case}")),
                case,
            );
        } else {
            let output = result.unwrap();
            check(
                &V::tensor(output.shallow_clone()),
                &case["output"],
                dev,
                "DEIS frame output",
            );
            assert_eq!(
                output.is_set_to(&x),
                case["returned_input"].as_bool().unwrap()
            );
        }
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "DEIS original caller sigma",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}

#[test]
fn original_stochastic_adams_and_pece_retain_prediction_lists_stacked_buffers_corrector_history_tau_frames_and_early_aliases()
 {
    use imagine_inference::sampling::adams;
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals5.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 264);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = Kind::Float;
        let x = Tensor::arange(12, (dtype, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3]);
        let values: Vec<f32> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let sigmas = Tensor::from_slice(&values).to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let model = Parameterization {
            prediction: if case["flow"] == true {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let tau =
            |_: &Tensor| -> Result<f64, Error> { Ok(if case["tau"] == "zero" { 0. } else { 0.5 }) };
        let output = adams::sample(
            &mut exec,
            x.shallow_clone(),
            &sigmas,
            model,
            adams::Settings {
                noise: case["noise"].as_f64().unwrap(),
                predictor: case["predictor"].as_i64().unwrap(),
                corrector: case["corrector"].as_i64().unwrap(),
                pece: case["name"] == "sa_solver_pece",
                simple_order_2: case["simple"] == true,
            },
            if case["tau"] == "default" {
                None
            } else {
                Some(&tau)
            },
        )
        .unwrap();
        check(
            &V::tensor(output.shallow_clone()),
            &case["output"],
            dev,
            "Adams/PECE frame output",
        );
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "Adams/PECE original caller sigma",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
        assert_eq!(
            output.is_set_to(&x),
            case["returned_input"].as_bool().unwrap()
        );
    }
}

#[test]
fn original_er_sde_preserves_integration_buffers_history_aliases_offsets_and_live_frames_during_custom_default_scalers()
 {
    use imagine_inference::sampling::exponential;
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals4.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 384);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas = Tensor::from_slice(&[
            case["first"].as_f64().unwrap() as f32,
            0.6,
            0.3,
            0.1,
            case["end"].as_f64().unwrap() as f32,
        ])
        .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let model = Parameterization {
            prediction: if case["flow"] == true {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let scaler = |value: &Tensor| -> Result<Tensor, Error> {
            Ok(value.f_pow_tensor_scalar(2)?.f_add_scalar(1.)?)
        };
        let output = exponential::er_sde(
            &mut exec,
            x,
            &sigmas,
            model,
            case["noise"].as_f64().unwrap(),
            case["stage"].as_i64().unwrap(),
            if case["custom"] == true {
                Some(&scaler)
            } else {
                None
            },
        )
        .unwrap();
        check(
            &V::tensor(output),
            &case["output"],
            dev,
            "ER-SDE frame output",
        );
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "ER-SDE original caller sigma",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}

#[test]
fn original_seeds_and_exponential_heun_retain_stochastic_stages_history_sigma_offsets_and_exact_tensor_storage_aliases()
 {
    use imagine_inference::sampling::exponential::{self, Seeds2, Seeds3};
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals3.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 384);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let options = &case["options"];
        let name = case["name"].as_str().unwrap();
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas = Tensor::from_slice(&[
            case["first"].as_f64().unwrap() as f32,
            0.6,
            0.3,
            0.1,
            case["end"].as_f64().unwrap() as f32,
        ])
        .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let model = Parameterization {
            prediction: if case["flow"] == true {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let output = if name == "seeds_3" {
            exponential::seeds_3(
                &mut exec,
                x,
                &sigmas,
                model,
                Seeds3 {
                    eta: options["eta"].as_f64().unwrap(),
                    noise: 0.875,
                    r1: 0.25,
                    r2: 0.75,
                },
            )
        } else {
            let deterministic = name == "exp_heun_2_x0";
            exponential::seeds_2(
                &mut exec,
                x,
                &sigmas,
                model,
                Seeds2 {
                    eta: if deterministic {
                        0.
                    } else {
                        options["eta"].as_f64().unwrap()
                    },
                    noise: if deterministic { 0. } else { 0.875 },
                    r: if name == "seeds_2" { 0.75 } else { 1. },
                },
                options["solver_type"].as_str().unwrap(),
            )
        }
        .unwrap();
        check(
            &V::tensor(output),
            &case["output"],
            dev,
            "SEEDS/Heun frame output",
        );
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "SEEDS/Heun original caller sigma",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}

#[test]
fn original_dpmpp_2s_and_res_multistep_preserve_eps_flow_cfgpp_special_sigmas_and_source_tensor_storages()
 {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals2.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 176);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas = Tensor::from_slice(&[
            case["first"].as_f64().unwrap() as f32,
            0.6,
            0.3,
            0.1,
            case["end"].as_f64().unwrap() as f32,
        ])
        .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let model = Parameterization {
            prediction: if case["flow"] == true {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let solver = if case["name"].as_str().unwrap().starts_with("res_multistep") {
            s::res_multistep
        } else {
            s::dpmpp_2s_ancestral
        };
        let output = solver(
            &mut exec,
            x,
            &sigmas,
            model,
            case["eta"].as_f64().unwrap(),
            0.875,
            case["name"].as_str().unwrap().ends_with("cfg_pp"),
        )
        .unwrap();
        check(
            &V::tensor(output),
            &case["output"],
            dev,
            "DPM++ 2S source-local output",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}

#[test]
fn source_sampler_locals_retain_exact_tensor_storages_through_model_and_callback_sites() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-locals.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 296);
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let sigmas = Tensor::from_slice(&[
            0.9f32,
            0.6,
            0.3,
            0.1,
            case["end"].as_f64().unwrap_or(0.) as f32,
        ])
        .to_device(dev);
        let mut exec = Exec {
            case,
            event: 0,
            rng: native.generator(dev, 17).unwrap(),
            initial_kind: dtype,
        };
        let name = case["name"].as_str().unwrap();
        let model = Parameterization {
            prediction: if case["flow"].as_bool().unwrap_or(false) {
                ParameterizationKind::Const
            } else {
                ParameterizationKind::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let result = if name == "ddpm" {
            s::ddpm(&mut exec, x, &sigmas)
        } else if name == "ipndm" {
            s::ipndm(&mut exec, x, &sigmas, case["advanced"].as_i64().unwrap())
        } else if name == "ipndm_v" {
            s::ipndm_v(&mut exec, x, &sigmas, case["advanced"].as_i64().unwrap())
        } else if name == "lms" {
            s::lms(
                &mut exec,
                x,
                &sigmas,
                case["advanced"].as_i64().unwrap(),
                &mut |message| panic!("unexpected fixture warning {message}"),
            )
        } else if name == "euler_ancestral" || name == "dpm_2_ancestral" {
            s::sample_ancestral(
                &mut exec,
                x,
                &sigmas,
                model,
                case["eta"].as_f64().unwrap(),
                0.875,
                if name == "euler_ancestral" {
                    s::AncestralSolver::Euler
                } else {
                    s::AncestralSolver::Dpm2
                },
            )
        } else if name == "euler_cfg_pp" || name == "euler_ancestral_cfg_pp" {
            s::euler_cfg_pp(
                &mut exec,
                x,
                &sigmas,
                model,
                if name == "euler_cfg_pp" {
                    0.
                } else {
                    case["eta"].as_f64().unwrap()
                },
                if name == "euler_cfg_pp" { 0. } else { 0.875 },
            )
        } else if name == "lcm" {
            let settings = if case["advanced"].as_bool().unwrap() {
                s::LcmSettings {
                    noise: 0.875,
                    noise_end: Some(0.25),
                    noise_clip_std: 1.5,
                }
            } else {
                s::LcmSettings::default()
            };
            s::lcm(&mut exec, x, &sigmas, model, settings)
        } else if name.starts_with("dpmpp_2m") {
            s::dpmpp_2m(&mut exec, x, &sigmas, name.ends_with("cfg_pp"))
        } else if name.starts_with("gradient_estimation") {
            s::gradient_estimation(&mut exec, x, &sigmas, 2., name.ends_with("cfg_pp"))
        } else {
            let solver = match name {
                "euler" => s::ChurnSolver::Euler,
                "heun" => s::ChurnSolver::Heun,
                "dpm_2" => s::ChurnSolver::Dpm2,
                "heunpp2" => s::ChurnSolver::HeunPP2,
                _ => unreachable!(),
            };
            s::churn(
                &mut exec,
                x,
                &sigmas,
                EulerSettings {
                    churn: case["churn"].as_f64().unwrap(),
                    tmin: 0.2,
                    tmax: 0.95,
                    noise: 0.875,
                },
                solver,
            )
        };
        let result = result.unwrap();
        check(
            &V::tensor(result),
            &case["output"],
            dev,
            "Source-local retention result",
        );
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}
