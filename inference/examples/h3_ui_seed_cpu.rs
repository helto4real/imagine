//! Actual AV prepare_noise with synthetic CPU tensors; no model or CUDA runtime.
use imagine_inference::{execution_value::Value, models::minimax_h3::streams::NestedTensor};
use serde_json::json;
use tch::{Device, Kind, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::set_num_threads(1);
    let mut cases = vec![];
    for seed in [0, 9_007_199_254_740_993, u64::MAX] {
        for kind in [Kind::Float, Kind::Half] {
            let inputs = NestedTensor::new(vec![
                Value::tensor(Tensor::zeros([1, 4, 3, 2, 2], (kind, Device::Cpu))),
                Value::tensor(Tensor::zeros([1, 2, 17], (kind, Device::Cpu))),
            ]);
            let noise = inputs.prepare_noise(seed, None)?;
            let tensors = noise.unbind();
            let streams=tensors.borrow().iter().map(|value| {
                let t=value.as_tensor().unwrap();
                json!({"shape":t.size(),"values":Vec::<f32>::try_from(t.to_kind(Kind::Float).view([-1])).unwrap()})
            }).collect::<Vec<_>>();
            cases.push(json!({"seed":seed,"dtype":if kind==Kind::Float{"float32"}else{"float16"},"streams":streams}));
        }
    }
    println!("{}", json!({"device":"cpu","cases":cases}));
    Ok(())
}
