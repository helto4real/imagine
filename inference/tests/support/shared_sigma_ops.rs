use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    models::qwen_image21::FluxSampling,
    native::{Error, storage_info},
    sampling::sigma_ops as s,
};
use serde_json::Value;
use tch::{Device, Tensor};
#[test]
fn shared_sigma_operations_preserve_output_input_views_and_parser_errors() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let previous = native.default_dtype().unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sigma-ops.json")).unwrap();
    let model = FluxSampling {
        shift: 0.69,
        sigmas: Tensor::new(),
    };
    for c in fixture["cases"].as_array().unwrap() {
        let dtype = kind(c["kind"].as_str().unwrap());
        native.set_default_dtype(dtype).unwrap();
        let dev = c["device"].as_str().map(device).unwrap_or(Device::Cpu);
        let values: Vec<f64> = c["values"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect())
            .unwrap_or_default();
        let input = Tensor::from_slice(&values).to_kind(dtype).to_device(dev);
        let args = &c["args"];
        let result: Result<Vec<Tensor>, Error> = match c["name"].as_str().unwrap() {
            "ManualSigmas" => s::manual(c["text"].as_str().unwrap()).map(|t| vec![t]),
            "SDTurboScheduler" => s::sd_turbo(
                &model,
                c["steps"].as_i64().unwrap(),
                c["denoise"].as_f64().unwrap(),
            )
            .map(|t| vec![t]),
            "SplitSigmas" => s::split(&input, args[0].as_i64().unwrap()).map(|(a, b)| vec![a, b]),
            "SplitSigmasDenoise" => {
                s::split_denoise(&input, args[0].as_f64().unwrap()).map(|(a, b)| vec![a, b])
            }
            "FlipSigmas" => s::flip(&input).map(|t| vec![t]),
            "SetFirstSigma" => s::set_first(&input, args[0].as_f64().unwrap()).map(|t| vec![t]),
            "ExtendIntermediateSigmas" => s::extend(
                &native,
                &input,
                args[0].as_i64().unwrap(),
                args[1].as_f64().unwrap(),
                args[2].as_f64().unwrap(),
                args[3].as_str().unwrap(),
            )
            .map(|t| vec![t]),
            _ => unreachable!(),
        };
        if c["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("Expected original error {c}")),
                c,
            );
        } else {
            let out = result.unwrap_or_else(|e| panic!("{c}: {e}"));
            assert_eq!(out.len(), c["output"].as_array().unwrap().len());
            for (i, t) in out.into_iter().enumerate() {
                let same = storage_info(&input).unwrap().0 == storage_info(&t).unwrap().0;
                if let Some(alias) = c["aliases"][i].as_bool() {
                    assert_eq!(same, alias, "{c}");
                }
                check(&V::tensor(t), &c["output"][i], dev, &format!("{c}"));
            }
        }
        if !c["input_after"].is_null() {
            check(
                &V::tensor(input),
                &c["input_after"],
                dev,
                "Sigma input mutation",
            );
        }
    }
    native.set_default_dtype(previous).unwrap();
}
