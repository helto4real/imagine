use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    native::{Error, Kitchen},
    sampling::{
        advanced_guidance as g,
        model_sampling::{Config, Flavor, ModelSampling},
    },
};
use serde_json::{Value, json};
use tch::{Device, Tensor};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0)
    }
}
fn compare(result: Result<V, Error>, case: &Value, dev: Device, note: &str) {
    if case["error"].is_string() {
        reference_error(
            result
                .err()
                .unwrap_or_else(|| panic!("Expected error: {note}")),
            case,
        )
    } else {
        check(
            &result.unwrap_or_else(|e| panic!("{note}: {e}")),
            &case["output"],
            dev,
            note,
        )
    }
}
fn svd(native: &Kitchen, x: &Tensor) -> Result<Tensor, Error> {
    let out = unsafe {
        imagine_inference::boxed_aten::call(
            native,
            "aten::linalg_svd",
            "",
            &[V::tensor(x.shallow_clone())],
            &vec![("full_matrices".into(), V::Bool(false))],
        )
    }?;
    let V::Tuple(out) = out else {
        panic!("ATen tuple")
    };
    out[2].as_tensor()
}
#[test]
fn built_in_cfg_norm_zero_star_tcfg_perp_neg_and_differential_masks_match_original() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(1);
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-advanced-guidance.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 1728);
    assert_eq!(fixture["masks"].as_array().unwrap().len(), 360);
    assert_eq!(fixture["fallback"].as_array().unwrap().len(), 4);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let args = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| restore(x, dev))
            .collect::<Vec<_>>();
        let t = |index: usize| args[index].as_tensor().unwrap();
        let scale = case["scale"].as_f64().unwrap_or(1.);
        let strength = case["strength"].as_f64().unwrap_or(1.);
        let result = match case["name"].as_str().unwrap() {
            "optimized_scale" => g::optimized_scale(&t(0), &t(1)).map(V::tensor),
            "tangential" => g::tangential_damping(&native, &t(0), &t(1)).map(V::tensor),
            "zero_star" => g::cfg_zero_star(&t(0), &t(1), &t(2), &t(3), scale).map(V::tensor),
            "norm_pre" => g::cfg_norm_pre(&t(0), &t(1), scale, strength).map(V::tensor),
            "norm_post" => g::cfg_norm_post(&t(0), &t(1), strength).map(V::tensor),
            "perp_neg" => g::perp_neg(
                &t(0),
                &t(1),
                &t(2),
                case["negative"].as_f64().unwrap(),
                scale,
            )
            .map(V::tensor),
            "tcfg" => {
                let V::List(original) = &args[1] else {
                    panic!("Own prediction list")
                };
                let values = original.clone();
                let address = values.as_ptr();
                let result = g::tcfg(
                    &native,
                    &t(0),
                    values,
                    [
                        case["present"][0].as_bool().unwrap(),
                        case["present"][1].as_bool().unwrap(),
                    ],
                );
                if let Ok(out) = &result {
                    assert_eq!(
                        out.as_ptr() == address,
                        case["same_list"].as_bool().unwrap()
                    );
                    assert_eq!(
                        json!(
                            out.iter()
                                .map(|v| original
                                    .iter()
                                    .map(|o| v.same_object(o))
                                    .collect::<Vec<_>>())
                                .collect::<Vec<_>>()
                        ),
                        case["identity"]
                    );
                }
                result.map(V::List)
            }
            _ => unreachable!(),
        };
        compare(
            result,
            case,
            dev,
            &format!("guidance {i} {} {}", case["name"], case["device"]),
        );
    }
    let model = ModelSampling::new(
        &native,
        Flavor::Flux,
        Config {
            shift: 0.69,
            timesteps: 10000,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    for (i, case) in fixture["masks"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let mask = restore(&case["mask"], dev).as_tensor().unwrap();
        let sigma = restore(&case["sigma"], Device::Cpu).as_tensor().unwrap();
        let sigmas = restore(&case["sigmas"], Device::Cpu).as_tensor().unwrap();
        compare(
            g::differential_mask(
                &model,
                &sigma,
                &mask,
                &sigmas,
                case["strength"].as_f64().unwrap(),
            )
            .map(V::tensor),
            case,
            dev,
            &format!("differential mask {i}"),
        );
    }
    for case in fixture["fallback"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let cond = restore(&case["args"][0], dev).as_tensor().unwrap();
        let uncond = restore(&case["args"][1], dev).as_tensor().unwrap();
        let mut calls = 0;
        let result = g::tangential_damping_with(&cond, &uncond, &mut |x| {
            check(
                &V::tensor(x.shallow_clone()),
                &case["calls"][calls],
                device(case["calls"][calls]["device"].as_str().unwrap()),
                "SVD fallback input",
            );
            calls += 1;
            if calls == 1 {
                return Err(Error::ReferenceException {
                    class: if case["failure"] == "RuntimeError" {
                        "RuntimeError"
                    } else {
                        "ValueError"
                    },
                    message: "own forced SVD boundary".into(),
                });
            }
            svd(&native, x)
        });
        assert_eq!(calls, case["calls"].as_array().unwrap().len());
        compare(result.map(V::tensor), case, dev, "SVD fallback");
    }
}
