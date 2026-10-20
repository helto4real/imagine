use super::{
    binding_tests::{check, reference_error, restore},
    kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    native::{Error, storage_info},
    sampling::{
        model_sampling::{Config, Flavor, ModelSampling},
        plan::{self, Plan, Selection, Settings},
        schedules::{self, ScheduleModel},
    },
};
use serde_json::{Value, json};
use std::cell::Cell;
use tch::{Device, Kind, Tensor};
struct Model {
    sampling: ModelSampling,
    calls: Cell<usize>,
}
impl plan::Model for Model {
    fn sampling(&self) -> Result<&dyn ScheduleModel, Error> {
        self.calls.set(self.calls.get() + 1);
        Ok(&self.sampling)
    }
}
struct Dtype<'a>(&'a imagine_inference::native::Kitchen, Kind);
impl Drop for Dtype<'_> {
    fn drop(&mut self) {
        self.0.set_default_dtype(self.1).unwrap();
    }
}
fn denoise(value: &Value) -> Option<f64> {
    match restore(value, Device::Cpu) {
        V::None => None,
        V::Float(x) => Some(x),
        _ => panic!("Own denoise value"),
    }
}
#[test]
fn original_ksampler_plans_and_sample_bounds_preserve_fallbacks_sigma_mutation_and_early_returns() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let _dtype = Dtype(&native, native.default_dtype().unwrap());
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-plan.json")).unwrap();
    assert_eq!(json!(&plan::SAMPLERS[..]), fixture["names"]["samplers"]);
    assert_eq!(json!(schedules::NAMES), fixture["names"]["schedulers"]);
    assert_eq!(fixture["plans"].as_array().unwrap().len(), 2850);
    assert_eq!(fixture["selections"].as_array().unwrap().len(), 1176);
    for kind_name in ["torch.float32", "torch.float64", "torch.bfloat16"] {
        let dtype = kind(kind_name);
        native.set_default_dtype(dtype).unwrap();
        let model = Model {
            sampling: ModelSampling::new(
                &native,
                Flavor::Flux,
                Config {
                    shift: 0.69,
                    timesteps: 10000,
                    ..Default::default()
                },
                None,
            )
            .unwrap(),
            calls: Cell::new(0),
        };
        for (index, case) in fixture["plans"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, c)| c["kind"] == kind_name)
        {
            model.calls.set(0);
            let mut warnings = Vec::new();
            let result = Plan::new(
                &model,
                &native,
                Settings {
                    steps: case["steps"].as_i64().unwrap(),
                    device: Device::Cpu,
                    sampler: case["sampler"].as_str(),
                    scheduler: case["scheduler"].as_str(),
                    denoise: denoise(&case["denoise"]),
                },
                &mut |m| warnings.push(m.to_owned()),
            );
            assert!(warnings.is_empty(), "Unexpected plan warnings {warnings:?}");
            assert_eq!(
                json!(vec!["model_sampling"; model.calls.get()]),
                case["events"],
                "plan {index}"
            );
            if case["error"].is_string() {
                reference_error(
                    result
                        .err()
                        .unwrap_or_else(|| panic!("Expected plan error {index}")),
                    case,
                );
                continue;
            }
            let out = result.unwrap_or_else(|e| panic!("plan {index}: {e}"));
            let expected = &case["output"];
            assert_eq!(out.sampler, expected["sampler"].as_str().unwrap());
            assert_eq!(out.scheduler, expected["scheduler"].as_str().unwrap());
            assert_eq!(json!(out.steps), expected["steps"]);
            check(
                &out.denoise.map_or(V::None, V::Float),
                &expected["denoise"],
                Device::Cpu,
                "stored denoise",
            );
            check(
                &V::tensor(out.sigmas),
                &expected["sigmas"],
                Device::Cpu,
                &format!("plan {index}"),
            );
        }
        for (index, case) in fixture["selections"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, c)| c["kind"] == kind_name)
        {
            let plan = Plan::new(
                &model,
                &native,
                Settings {
                    steps: 7,
                    device: Device::Cpu,
                    sampler: Some("euler"),
                    scheduler: Some("simple"),
                    denoise: None,
                },
                &mut |m| panic!("Unexpected warning {m}"),
            )
            .unwrap();
            let noise = Tensor::arange(24, (dtype, Device::Cpu))
                .reshape([1, 2, 3, 4])
                .f_div_scalar(32)
                .unwrap();
            let caller = case["override"]
                .as_bool()
                .unwrap()
                .then(|| Tensor::from_slice(&[1., 0.5, 0.]).to_kind(dtype));
            let latent = case["latent"]
                .as_bool()
                .unwrap()
                .then(|| V::tensor(Tensor::full_like(&noise, 0.375)));
            let result = plan.select(
                &noise,
                latent.as_ref(),
                caller.as_ref(),
                case["start"].as_i64(),
                case["last"].as_i64(),
                case["force"].as_bool().unwrap(),
            );
            if case["error"].is_string() {
                reference_error(
                    result
                        .err()
                        .unwrap_or_else(|| panic!("Expected selection error {index}")),
                    case,
                );
            } else {
                match result.unwrap_or_else(|e| panic!("selection {index}: {e}")) {
                    Selection::Early(value) => {
                        assert!(case["calls"].as_array().unwrap().is_empty());
                        assert!(case["factories"].as_array().unwrap().is_empty());
                        assert_eq!(
                            latent.as_ref().is_some_and(|v| v.same_object(&value)),
                            case["output"]["latent_same"].as_bool().unwrap()
                        );
                        check(
                            &value,
                            &case["output"]["value"],
                            Device::Cpu,
                            &format!("early selection {index}"),
                        );
                    }
                    Selection::Sampling(sigmas) => {
                        assert_eq!(case["factories"], json!(["euler"]));
                        assert_eq!(case["calls"].as_array().unwrap().len(), 1);
                        let call = &case["calls"][0];
                        assert_eq!(call["sampler"], "euler");
                        assert_eq!(
                            storage_info(&sigmas).unwrap().0
                                == storage_info(&plan.sigmas).unwrap().0,
                            call["self_storage"].as_bool().unwrap()
                        );
                        assert_eq!(
                            caller
                                .as_ref()
                                .is_some_and(|c| storage_info(c).unwrap().0
                                    == storage_info(&sigmas).unwrap().0),
                            call["caller_storage"].as_bool().unwrap()
                        );
                        check(
                            &V::tensor(sigmas),
                            &call["sigmas"],
                            Device::Cpu,
                            &format!("selection {index}"),
                        );
                    }
                }
            }
            check(
                &V::tensor(plan.sigmas),
                &case["self_after"],
                Device::Cpu,
                &format!("plan post {index}"),
            );
            check(
                &caller.map_or(V::None, V::tensor),
                &case["caller_after"],
                Device::Cpu,
                &format!("caller post {index}"),
            );
        }
    }
}
