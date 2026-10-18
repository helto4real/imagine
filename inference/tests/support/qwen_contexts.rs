use super::kitchen;
use imagine_inference::{
    models::qwen_image21 as q,
    native::{self, Error, Generator},
};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};

fn snapshot(base: &Tensor, label: &str) -> Value {
    let output = base * 2.;
    json!({"label":label,"requires_grad":output.requires_grad(),"inference":output.is_inference()})
}
struct Execution {
    generator: Generator,
    fail: bool,
    calls: Vec<Value>,
}
impl q::EulerExecution for Execution {
    fn denoise(&mut self, x: &Tensor, _sigma: &Tensor) -> Result<Tensor, Error> {
        Ok(x.f_mul_scalar(0.25)?)
    }
    fn noise_like(&mut self, x: &Tensor) -> Result<Tensor, Error> {
        self.generator.randn(&x.size(), x.kind())
    }
    fn callback(&mut self, step: q::EulerStep<'_>) -> Result<(), Error> {
        self.calls.push(json!({"requires_grad":step.denoised.requires_grad(),"inference":step.denoised.is_inference()}));
        if self.fail {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "injected observer exit".into(),
            });
        }
        Ok(())
    }
}
#[test]
fn qwen_image21_original_execution_contexts_and_sampler_grad_state_match() {
    let library = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-contexts.json")).unwrap();
    let base = Tensor::ones([1, 1, 1, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let mut records = vec![snapshot(&base, "normal")];
    {
        let _on = native::inference_mode(true).unwrap();
        records.push(snapshot(&base, "on"));
        {
            let _off = native::inference_mode(false).unwrap();
            records.push(snapshot(&base, "disabled"));
        }
        records.push(snapshot(&base, "on-restored"));
        let result: Result<(), Error> = (|| {
            let _off = native::inference_mode(false)?;
            records.push(snapshot(&base, "disabled-before-error"));
            Err(Error::Native("own bounded scope exit".into()))
        })();
        assert!(result.is_err());
        records.push(snapshot(&base, "on-after-error"));
    }
    records.push(snapshot(&base, "normal-restored"));
    {
        let _grad = tch::no_grad_guard();
        records.push(snapshot(&base, "no-grad"));
        {
            let _on = native::inference_mode(true).unwrap();
            records.push(snapshot(&base, "nested-on"));
            {
                let _off = native::inference_mode(false).unwrap();
                records.push(snapshot(&base, "nested-off"));
            }
            records.push(snapshot(&base, "nested-restored"));
        }
        records.push(snapshot(&base, "no-grad-restored"));
    }
    records.push(snapshot(&base, "normal-final"));
    assert_eq!(json!(records), fixture["contexts"]);
    for case in fixture["samplers"].as_array().unwrap() {
        let mut execution = Execution {
            generator: library.generator(Device::Cpu, 18).unwrap(),
            fail: case["fail"].as_bool().unwrap(),
            calls: vec![],
        };
        let sigmas = Tensor::from_slice(&[0.8f32, 0.4, 0.]);
        let result = if case["method"] == "sample_euler" {
            q::sample_euler(
                base.shallow_clone(),
                &sigmas,
                Default::default(),
                &mut execution,
            )
        } else {
            q::sample_er_sde(
                base.shallow_clone(),
                &sigmas,
                &q::FluxSampling::qwen21().unwrap(),
                q::ErSdeSettings {
                    noise: 0.,
                    ..Default::default()
                },
                &mut execution,
            )
        };
        assert_eq!(json!(execution.calls), case["calls"]);
        if let Some(expected) = case.get("error") {
            let Error::ReferenceException { class, message } = result.unwrap_err() else {
                panic!("expected source callback failure")
            };
            assert_eq!(class, expected["class_name"].as_str().unwrap());
            assert_eq!(message, expected["message"].as_str().unwrap());
        } else {
            let output = result.unwrap();
            assert_eq!(
                json!({"requires_grad":output.requires_grad(),"inference":output.is_inference()}),
                case["output"]
            );
        }
        assert_eq!(json!((&base * 2.).requires_grad()), case["restored_grad"]);
    }
}
