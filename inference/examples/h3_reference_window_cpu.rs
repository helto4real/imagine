//! Synthetic CPU attention only; no Runtime, GPU, weights or Exp diagnostics.
use imagine_inference::{
    h3_product::{protocol::Shape, reference_window},
    models::minimax_h3::layout::ReferenceBlock,
};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::set_num_threads(1);
    let inputs: Vec<Value> = serde_json::from_reader(std::io::stdin())?;
    let mut outputs = vec![];
    for input in inputs {
        let shape: Shape = serde_json::from_value(input["shape"].clone())?;
        let refs: Vec<ReferenceBlock> = serde_json::from_value(input["refs"].clone())?;
        let n = input["n"].as_i64().unwrap();
        let q = Tensor::f_arange(n * 2, (Kind::Float, Device::Cpu))?
            .f_view([1, 1, n, 2])?
            .f_div_scalar(1000.)?;
        let k = q.f_flip([2])?;
        let v = q.f_sin()?;
        let window = reference_window::prepare(
            shape,
            input["text"].as_i64().unwrap(),
            input["keyframe_rows"].as_i64().unwrap(),
            &refs,
        )?;
        let out = window.run(&q, &k, &v, |q, k, v| {
            Ok(q.f_matmul(&k.f_transpose(-2, -1)?)?
                .f_div_scalar(2f64.sqrt())?
                .f_softmax(-1, Kind::Float)?
                .f_matmul(v)?
                .f_transpose(1, 2)?
                .f_contiguous()?
                .f_view([1, -1, 2])?)
        })?;
        outputs.push(json!(Vec::<f32>::try_from(&out.f_view([-1])?)?));
    }
    println!("{}", json!(outputs));
    Ok(())
}
