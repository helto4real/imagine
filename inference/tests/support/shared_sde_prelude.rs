use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    models::qwen_image21::EulerStep,
    native::Error,
    sampling::{
        brownian::BrownianNoise,
        parameterization::{Parameterization, Prediction as P},
        sde::{self, Source},
        solvers::{Execution, NoiseKind, Prediction},
    },
};
use serde_json::Value;
use tch::{Kind, Tensor};
struct Exec<'a> {
    case: &'a Value,
    event: usize,
}
impl Exec<'_> {
    fn event(&mut self, op: &str) -> &Value {
        let event = &self.case["events"][self.event];
        assert_eq!(event["op"], op);
        self.event += 1;
        event
    }
}
impl Execution for Exec<'_> {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, _cfg: bool) -> Result<Prediction, Error> {
        let expected = self.event("model");
        check(
            &V::tensor(x.shallow_clone()),
            &expected["x"],
            x.device(),
            "prelude model x",
        );
        check(
            &V::tensor(sigma.shallow_clone()),
            &expected["sigma"],
            x.device(),
            "prelude model sigma",
        );
        Err(Error::ReferenceException {
            class: "ValueError",
            message: "own SDE model stop".into(),
        })
    }
    fn make_brownian(
        &mut self,
        x: &Tensor,
        min: &Tensor,
        max: &Tensor,
        cpu: bool,
    ) -> Result<BrownianNoise, Error> {
        let expected = self.event("brownian");
        assert_eq!(cpu, expected["cpu"] == true);
        assert_eq!(expected["seed"], 901);
        for (name, value) in [("x", x), ("minimum", min), ("maximum", max)] {
            check(
                &V::tensor(value.shallow_clone()),
                &expected[name],
                x.device(),
                "prelude Brownian arguments",
            );
        }
        Err(Error::ReferenceException {
            class: "ValueError",
            message: "own Brownian constructor stop".into(),
        })
    }
    fn noise(&mut self, _: &Tensor, _: &Tensor, _: &Tensor, _: NoiseKind) -> Result<Tensor, Error> {
        panic!("unreachable prelude noise")
    }
    fn callback(&mut self, _: EulerStep<'_>) -> Result<(), Error> {
        panic!("unreachable prelude callback")
    }
}
#[test]
fn original_sde_preludes_preserve_scalar_length_errors_unconditional_bounds_custom_gpu_wrappers_constructor_order_and_invalid_solver_precedence()
 {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sde-prelude.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 768);
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
        let sigmas = match case["mode"].as_str().unwrap() {
            "empty" => Tensor::from_slice(&[] as &[f32]),
            "one" => Tensor::from_slice(&[0.9f32]),
            "scalar" => Tensor::from(0.9f32),
            "zero" => Tensor::from_slice(&[0f32, 0.]),
            "negative" => Tensor::from_slice(&[-0.3f32, 0.]),
            "positive" => Tensor::from_slice(&[0.9f32, 0.3, 0.]),
            "ambiguous" => Tensor::from_slice(&[0.9f32, 0.6, 0.2, 0.]).reshape([2, 2]),
            "nan" => Tensor::from_slice(&[0.9f32, f32::NAN, 0.]),
            _ => panic!("own schedule"),
        }
        .to_device(dev);
        let source = match (case["custom"] == true, name.ends_with("_gpu")) {
            (true, true) => Source::CustomGpu,
            (true, false) => Source::Custom,
            (false, true) => Source::Gpu,
            (false, false) => Source::Cpu,
        };
        let settings = sde::Settings {
            eta: 1.,
            noise: 1.,
            source,
        };
        let model = Parameterization {
            prediction: P::Eps,
            sigma_data: 1.,
            noise_scale: 1.,
        };
        let mut exec = Exec { case, event: 0 };
        let result = if name.starts_with("dpmpp_2m") {
            sde::dpmpp_2m_sde(
                &mut exec,
                x.shallow_clone(),
                &sigmas,
                model,
                settings,
                if case["invalid"] == true {
                    "invalid"
                } else if name.contains("heun") {
                    "heun"
                } else {
                    "midpoint"
                },
            )
        } else if name.starts_with("dpmpp_3m") {
            sde::dpmpp_3m_sde(&mut exec, x.shallow_clone(), &sigmas, model, settings)
        } else {
            sde::dpmpp_sde(&mut exec, x.shallow_clone(), &sigmas, model, settings, 0.5)
        };
        if case["error"].is_string() {
            reference_error(
                result.err().unwrap_or_else(|| panic!("expected {case}")),
                case,
            );
        } else {
            let output = result.unwrap();
            check(
                &V::tensor(output.shallow_clone()),
                &case["output"],
                dev,
                "SDE prelude early output",
            );
            assert_eq!(
                output.is_set_to(&x),
                case["returned_input"].as_bool().unwrap()
            );
        }
        assert_eq!(
            exec.event,
            case["events"].as_array().unwrap().len(),
            "events {case}"
        );
    }
}
