use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{execution_value::Value as V, models::qwen_image21 as q, native::Error};
use serde_json::{Value, json};
use std::cell::Cell;
use tch::{Device, Kind, Tensor};

fn data(shape: &[i64], dev: Device, dtype: Kind, offset: i64) -> Tensor {
    (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, dev)) + offset)
        .remainder(29)
        .g_mul_scalar(0.03125)
        .g_sub_scalar(0.40625)
        .reshape(shape)
        .to_kind(dtype)
}
struct Execution<'a> {
    case: &'a Value,
    step: usize,
    frees: usize,
    estimates: Cell<usize>,
    dev: Device,
}
impl q::FullAreaExecution for Execution<'_> {
    fn free_memory(&mut self, device: Device) -> Result<f64, Error> {
        assert_eq!(device, self.dev);
        assert_eq!(
            self.case["free_calls"][self.frees].as_str().unwrap(),
            if device == Device::Cpu {
                "cpu"
            } else {
                "cuda:0"
            }
        );
        self.frees += 1;
        Ok(self.case["free"].as_f64().unwrap())
    }
    fn memory_required(&self, input: &[i64], conditions: &[&q::FullAreaCondition]) -> f64 {
        let expected = &self.case["memory_calls"][self.estimates.get()];
        assert_eq!(json!(input), expected["shape"]);
        let shapes = json!({
            "c_crossattn":conditions.iter().map(|c|c.context.size()).collect::<Vec<_>>(),
            "attention_mask":conditions.iter().map(|c|c.attention_mask.as_ref().unwrap().size()).collect::<Vec<_>>(),
            "ref_latents":conditions.iter().map(|c|vec![1,64,c.reference_latents.as_ref().unwrap().iter().map(|r|r.numel() as i64).sum::<i64>()/64]).collect::<Vec<_>>(),
            "image_slots":vec![vec![1];conditions.len()]
        });
        assert_eq!(shapes, expected["cond_shapes"]);
        self.estimates.set(self.estimates.get() + 1);
        100. * input[0] as f64 / (2 * self.case["batch"].as_i64().unwrap()) as f64
    }
    fn apply_model(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        condition: &q::FullAreaCondition,
        branches: &[usize],
    ) -> Result<Tensor, Error> {
        let expected = &self.case["calls"][self.step];
        self.step += 1;
        assert_eq!(json!(branches), expected["branches"]);
        assert_eq!(json!(condition.image_slots), expected["slots"]);
        for (key, tensor) in [
            ("x", x),
            ("time", timestep),
            ("context", &condition.context),
            ("mask", condition.attention_mask.as_ref().unwrap()),
        ] {
            check(
                &V::tensor(tensor.shallow_clone()),
                &expected[key],
                self.dev,
                key,
            );
        }
        let refs = condition.reference_latents.as_ref().unwrap();
        for (reference, expected) in refs.iter().zip(expected["refs"].as_array().unwrap()) {
            check(
                &V::tensor(reference.shallow_clone()),
                expected,
                self.dev,
                "CFG repeated reference tensor",
            );
        }
        let context = condition.context.select(1, 0).select(1, 0);
        let reference = refs[0].select(1, 0).select(1, 0).select(1, 0);
        Ok(x.f_mul_scalar(0.125)?
            .f_add(&context.f_reshape([-1, 1, 1, 1])?.f_mul_scalar(0.25)?)?
            .f_add(&reference.f_reshape([-1, 1, 1, 1])?.f_mul_scalar(0.0625)?)?)
    }
}
#[test]
fn qwen_image21_original_full_area_cfg_condition_batches_memory_and_order_match() {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-cfg.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["dtype"].as_str().unwrap());
        let batch = case["batch"].as_i64().unwrap();
        let cond_batch = case["cond_batch"].as_i64().unwrap();
        let condition = |index: i64| {
            let tokens = if index == 0 {
                3
            } else {
                case["tokens"].as_i64().unwrap()
            };
            q::FullAreaCondition {
                context: data(&[cond_batch, tokens, 4], dev, dtype, 7 + index * 4),
                attention_mask: Some(data(&[cond_batch, tokens], dev, dtype, 13 + index * 4)),
                reference_latents: Some(vec![data(
                    &[cond_batch, 64, 2, 3],
                    dev,
                    dtype,
                    19 + index * 4,
                )]),
                image_slots: Some(vec![if case["slots_differ"] == true {
                    index
                } else {
                    0
                }]),
                concat: None,
            }
        };
        let positive = condition(0);
        let negative = condition(1);
        let mut execution = Execution {
            case,
            step: 0,
            frees: 0,
            estimates: Cell::new(0),
            dev,
        };
        let output = q::full_area_cfg(
            &data(&[batch, 64, 2, 3], dev, dtype, 3),
            &Tensor::full([batch], 0.625, (Kind::Float, dev)),
            &positive,
            Some(&negative),
            case["cfg"].as_f64().unwrap(),
            case["disabled"].as_bool().unwrap(),
            &mut execution,
        );
        if case["error"].is_object() {
            let Error::ReferenceException { class, message } = output.unwrap_err() else {
                panic!("Expected original empty-condition failure")
            };
            assert_eq!(class, case["error"]["class_name"].as_str().unwrap());
            assert_eq!(message, case["error"]["message"].as_str().unwrap());
        } else {
            check(
                &V::tensor(output.unwrap()),
                &case["output"],
                dev,
                "Original CFG output",
            );
        }
        assert_eq!(execution.step, case["calls"].as_array().unwrap().len());
        assert_eq!(
            execution.frees,
            case["free_calls"].as_array().unwrap().len()
        );
        assert_eq!(
            execution.estimates.get(),
            case["memory_calls"].as_array().unwrap().len()
        );
    }
}
