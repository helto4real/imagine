use super::{
    binding_tests::{check, reference_error},
    device, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    models::qwen_image21::{EulerStep, FluxSampling},
    native::{Error, Generator, Kitchen, WeakStorage},
    sampling::{
        brownian::{BrownianNoise, Seeds},
        default_call,
        locals::SourceLocals,
        parameterization::{Parameterization, Prediction as P},
        solvers::{self, Execution, NoiseKind, Prediction},
    },
};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};
struct Exec<'a> {
    native: &'a Kitchen,
    initial: WeakStorage,
    scaled: Option<WeakStorage>,
    events: Vec<Value>,
    global: Generator,
    independent: Generator,
    warnings: Vec<String>,
}
impl Exec<'_> {
    fn observe(&mut self, op: &str) {
        self.events.push(json!({"op":op,"initial":!self.initial.expired(),"scaled":self.scaled.as_ref().map(|value|!value.expired())}));
    }
    fn predict(&mut self, x: &Tensor, sigma: &Tensor, capture: bool) -> Result<Prediction, Error> {
        self.observe("model");
        let y = x
            .f_mul_scalar(0.125)?
            .f_add(&sigma.f_reshape([2, 1, 1, 1])?.f_mul_scalar(0.25)?)?;
        let uncond = if capture {
            Some(y.f_mul_scalar(0.5)?.f_sub_scalar(0.0625)?)
        } else {
            None
        };
        let cond = if capture {
            Some(y.f_mul_scalar(1.25)?.f_add_scalar(0.03125)?)
        } else {
            None
        };
        Ok(Prediction {
            denoised: y,
            uncond_denoised: uncond,
            cond_denoised: cond,
        })
    }
}
impl Execution for Exec<'_> {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg: bool) -> Result<Prediction, Error> {
        self.predict(x, sigma, cfg)
    }
    fn denoise_with_capture(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        _disable: bool,
    ) -> Result<Prediction, Error> {
        self.predict(x, sigma, true)
    }
    fn noise(
        &mut self,
        x: &Tensor,
        _: &Tensor,
        _: &Tensor,
        kind: NoiseKind,
    ) -> Result<Tensor, Error> {
        if kind == NoiseKind::GlobalLike {
            self.global.randn(&x.size(), x.kind())
        } else {
            self.independent.randn(&[2, 1, 2, 3], Kind::Float)
        }
    }
    fn callback(&mut self, _: EulerStep<'_>) -> Result<(), Error> {
        self.observe("callback");
        Ok(())
    }
    fn callback_values(&mut self, _: &[(String, V)]) -> Result<(), Error> {
        self.observe("callback");
        Ok(())
    }
    fn source_locals(&mut self, frame: &SourceLocals, event: &str) -> Result<(), Error> {
        if event == "unipc_outer" {
            let noise = frame
                .tensors()
                .find(|(name, _)| *name == "noise")
                .unwrap()
                .1;
            self.scaled = Some(WeakStorage::new(noise)?);
            self.observe("scaled");
        }
        Ok(())
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
            Seeds::One(901),
            cpu,
            Box::new(|value| Ok(value.shallow_clone())),
        )
    }
    fn percent_to_sigma(&self, percent: f64) -> Result<f64, Error> {
        Ok(FluxSampling {
            shift: 0.69,
            sigmas: Tensor::new(),
        }
        .percent_to_sigma(percent))
    }
    fn warning(&mut self, message: &str) {
        self.warnings.push(message.to_string());
    }
}
#[test]
fn original_default_kernels_preserve_decorator_arguments_undecorated_aliases_unipc_rebinding_temporary_input_storage_release_outputs_and_rng()
 {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-sampler-argument-roots.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 180);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
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
            .reshape([2, 1, 2, 3]);
        let initial = WeakStorage::new(&x).unwrap();
        let sigmas = Tensor::from_slice(&[0.9f32, 0.6, 0.3, 0.1, 0.]).to_device(dev);
        let mut exec = Exec {
            native: &native,
            initial,
            scaled: None,
            events: vec![],
            global: native.generator(dev, 17).unwrap(),
            independent: native
                .generator(dev, if dev == Device::Cpu { 902 } else { 901 })
                .unwrap(),
            warnings: vec![],
        };
        let model = Parameterization {
            prediction: P::Eps,
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        let mut warnings = vec![];
        let result = if name == "_cfgpp_history" {
            solvers::cfgpp_history(
                &mut exec,
                x,
                &sigmas,
                model,
                solvers::CfgHistory {
                    history_weight: 0.5,
                    zero_weight: None,
                    zero_order: 1,
                    uncond_history_weight: 0.,
                },
                &mut |message| warnings.push(message.to_string()),
            )
        } else {
            default_call::sample(
                name,
                &mut exec,
                &native,
                x,
                &sigmas,
                model,
                &mut |message| warnings.push(message.to_string()),
            )
        };
        exec.observe("returned");
        if case["error"].is_string() {
            reference_error(
                result.err().unwrap_or_else(|| panic!("expected {case}")),
                case,
            );
        } else {
            let output = result.unwrap_or_else(|error| panic!("{error:?}: {case}"));
            check(
                &V::tensor(output.shallow_clone()),
                &case["output"],
                dev,
                "default kernel output",
            );
            drop(output);
        }
        exec.observe("output-dropped");
        warnings.append(&mut exec.warnings);
        assert_eq!(
            json!(exec.events),
            case["events"],
            "argument roots {index} {name} {case}"
        );
        assert_eq!(json!(warnings), case["warnings"], "warnings {name}");
        check(
            &V::tensor(sigmas),
            &case["post_sigmas"],
            dev,
            "default kernel caller sigma",
        );
        check(
            &V::tensor(exec.global.state().unwrap()),
            &case["state"],
            dev,
            "default kernel RNG",
        );
    }
}
