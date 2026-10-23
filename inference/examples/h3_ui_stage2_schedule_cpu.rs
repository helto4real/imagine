//! Synthetic CPU comparison target for Director's two-stage schedule composition.
//! The callback curves are supplied by the test, not generated from model weights.
use imagine_inference::h3_product::{protocol::sampling::Sampling, sampling};
use serde_json::{Value, json};
use tch::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::set_num_threads(1);
    let input: Value = serde_json::from_slice(&std::fs::read(
        std::env::args()
            .nth(1)
            .ok_or("Expected own synthetic case file")?,
    )?)?;
    let mut cases = vec![];
    for case in input["cases"].as_array().ok_or("Expected case list")? {
        let spec: Sampling = serde_json::from_value(case["sampling"].clone())?;
        let mut calls = vec![];
        let result = sampling::resolve(&spec, |stage| {
            let source = if stage.scheduler == "simple" {
                "first"
            } else {
                "second"
            };
            calls.push(json!({"source":source,"scheduler":stage.scheduler,
                "steps":stage.steps,"denoise":stage.denoise}));
            let curve: Vec<f32> =
                serde_json::from_value(case[format!("{source}_curve")].clone())
                    .map_err(|e| imagine_inference::native::Error::Native(e.to_string()))?;
            Ok(Tensor::f_from_slice(&curve)?)
        });
        let result = match result {
            Ok(stages) => {
                let stages = stages.iter().map(|t| -> Result<Value, tch::TchError> {
                    let values=Vec::<f32>::try_from(t)?;
                    Ok(json!({"values":values,"bits":values.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}))
                }).collect::<Result<Vec<_>,_>>()?;
                json!({"stages":stages})
            }
            Err(error) => json!({"error":error.to_string()}),
        };
        cases.push(json!({"name":case["name"],"calls":calls,"result":result}));
    }
    println!("{}", json!({"device":"cpu","cases":cases}));
    Ok(())
}
