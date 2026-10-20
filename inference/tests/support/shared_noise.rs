use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    sampling::{noise, quadrature},
};
use serde_json::{Value, json};
use tch::Tensor;
#[test]
fn default_noise_closure_preserves_source_storage_live_metadata_mutation_and_global_rng() {
    use imagine_inference::native::{WeakStorage, global_rng_state};
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-noise-owners.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 48);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let mut x = Tensor::zeros([2, 3], (dtype, dev));
        tch::manual_seed(17);
        tch::Cuda::manual_seed_all(17);
        let weak = WeakStorage::new(&x).unwrap();
        let mut sampler = Some(
            noise::DefaultNoise::new(&native, &x, case["seed"].as_i64().map(i128::from)).unwrap(),
        );
        let mut events = vec![];
        let observe =
            |events: &mut Vec<Value>, name: &str| events.push(json!([name, !weak.expired()]));
        observe(&mut events, "constructed");
        let mutation = (|| -> Result<(), imagine_inference::native::Error> {
            match case["style"].as_str().unwrap() {
                "resize" => {
                    let _ = x.f_resize_([3, 2])?;
                }
                "grow" => {
                    let _ = x.f_resize_([5, 3])?;
                }
                "set_dtype" => {
                    let other = if dtype == tch::Kind::Float {
                        tch::Kind::BFloat16
                    } else {
                        tch::Kind::Float
                    };
                    x.f_set_data(&x.f_to_kind(other)?)?;
                }
                "set_device" => {
                    let other = if dev == tch::Device::Cpu {
                        tch::Device::Cuda(0)
                    } else {
                        tch::Device::Cpu
                    };
                    x.f_set_data(&x.f_to_device(other)?)?;
                }
                "transpose" => x = x.f_transpose(0, 1)?,
                "drop" => {}
                _ => unreachable!(),
            };
            Ok(())
        })();
        if case["mutation"]["error"].is_string() {
            reference_error(mutation.err().unwrap(), &case["mutation"]);
        } else {
            mutation.unwrap();
        }
        observe(&mut events, "mutated");
        drop(x);
        observe(&mut events, "input_dropped");
        for (draw, expected) in case["outputs"].as_array().unwrap().iter().enumerate() {
            let result = sampler.as_mut().unwrap().sample();
            if expected["error"].is_string() {
                reference_error(
                    result
                        .err()
                        .unwrap_or_else(|| panic!("Expected error {index}/{draw}")),
                    expected,
                );
            } else {
                let actual = result.unwrap_or_else(|e| panic!("Noise owner {index}/{draw}: {e}"));
                let output_device = actual.device();
                check(
                    &V::tensor(actual),
                    &expected["output"],
                    output_device,
                    &format!("live noise metadata {index}/{draw}"),
                );
            }
            observe(&mut events, &format!("draw_{draw}"));
        }
        check(
            &V::tensor(global_rng_state(tch::Device::Cpu).unwrap()),
            &case["state_cpu"],
            tch::Device::Cpu,
            "source noise CPU RNG",
        );
        check(
            &V::tensor(global_rng_state(tch::Device::Cuda(0)).unwrap()),
            &case["state_cuda"],
            tch::Device::Cpu,
            "source noise CUDA RNG",
        );
        drop(sampler.take());
        observe(&mut events, "sampler_dropped");
        assert_eq!(
            json!(events),
            case["events"],
            "closure lifetime {index} {}",
            case["style"]
        );
    }
}
#[test]
fn shared_numpy_seed_sequences_lms_coefficients_and_noise_streams_match_original() {
    let native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-noise-quadrature.json")).unwrap();
    for case in fixture["seed_cases"].as_array().unwrap() {
        let entropy = noise::words(case["entropy"].as_str().unwrap().parse::<u128>().unwrap());
        let spawn = case["spawn"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| noise::words(s.as_str().unwrap().parse::<u128>().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            json!(
                noise::seed_sequence(
                    &entropy,
                    &spawn,
                    case["pool"].as_u64().unwrap() as usize,
                    16
                )
                .unwrap()
            ),
            case["state"],
            "{case}"
        );
    }
    for case in fixture["coefficients"].as_array().unwrap() {
        let times = Tensor::from_slice(&[1., 0.75, 0.3, 0.125, 0.05, 0.])
            .to_kind(kind(case["kind"].as_str().unwrap()));
        let out = quadrature::coefficient(
            case["order"].as_u64().unwrap() as usize,
            &times,
            case["i"].as_u64().unwrap() as usize,
            case["j"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let bits = out
            .value
            .to_le_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            bits,
            case["bits"].as_str().unwrap(),
            "QAGSE+NumPy scalar coefficient {case}, native {out:?}"
        );
        let warnings = out
            .warning()
            .map(|message| vec![json!({"category":"IntegrationWarning","message":message})])
            .unwrap_or_default();
        assert_eq!(json!(warnings), case["warnings"], "{case}");
    }
    for case in fixture["noises"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let x = Tensor::zeros([2, 1, 2, 3], (kind(case["kind"].as_str().unwrap()), dev));
        let result = noise::DefaultNoise::new(
            &native,
            &x,
            Some(case["seed"].as_str().unwrap().parse::<i128>().unwrap()),
        );
        if case["error"].is_string() {
            reference_error(result.err().unwrap(), case);
        } else {
            let mut noise = result.unwrap();
            for expected in case["outputs"].as_array().unwrap() {
                check(
                    &V::tensor(noise.sample().unwrap()),
                    expected,
                    dev,
                    "independent stream",
                );
            }
            assert!(noise.state().unwrap().is_some());
        }
    }
    let invalid = noise::seed_sequence(&[1], &[], 3, 4).unwrap_err();
    assert!(matches!(
        invalid,
        imagine_inference::native::Error::ReferenceException {
            class: "ValueError",
            ..
        }
    ));
    let source_unsupported = quadrature::coefficient(
        1,
        &Tensor::from_slice(&[1., 0.]).to_kind(tch::Kind::BFloat16),
        0,
        0,
    )
    .unwrap_err();
    assert!(matches!(
        source_unsupported,
        imagine_inference::native::Error::ReferenceException {
            class: "TypeError",
            ..
        }
    ));
}

#[test]
fn shared_brownian_default_trees_noise_cache_and_buffer_ownership_match_original() {
    use imagine_inference::sampling::brownian::{BrownianNoise, Seeds, Transform};
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-brownian.json")).unwrap();
    for (case_index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let x = Tensor::zeros([2, 1, 2, 3], (kind(case["kind"].as_str().unwrap()), dev));
        let seeds = if let Some(seeds) = case["seed"].as_array() {
            Seeds::Batch(seeds.iter().map(|s| s.as_u64().unwrap()).collect())
        } else {
            Seeds::One(case["seed"].as_u64().unwrap())
        };
        let transform: Transform = if case["transform"] == "identity" {
            Box::new(|x| Ok(x.shallow_clone()))
        } else {
            Box::new(|x| Ok(x.f_log()?.f_neg()?))
        };
        let mut noise = BrownianNoise::new(
            &native,
            &x,
            &Tensor::from(0.05f32).to_device(dev),
            &Tensor::from(0.95f32).to_device(dev),
            seeds,
            case["cpu"].as_bool().unwrap(),
            transform,
        )
        .unwrap();
        for (query_index, query) in case["queries"].as_array().unwrap().iter().enumerate() {
            let mut warnings = Vec::new();
            let output = noise
                .sample(
                    &Tensor::from(query["start"].as_f64().unwrap() as f32).to_device(dev),
                    &Tensor::from(query["end"].as_f64().unwrap() as f32).to_device(dev),
                    &mut |m| warnings.push(m.to_owned()),
                )
                .unwrap();
            check(
                &V::tensor(output),
                &query["output"],
                dev,
                &format!(
                    "Brownian tree {case_index} query {query_index}: {} {} cpu={} seed={} transform={}",
                    case["device"], case["kind"], case["cpu"], case["seed"], case["transform"]
                ),
            );
            assert_eq!(
                noise.buffer_bytes().unwrap(),
                query["buffers"].as_u64().unwrap(),
                "source retained storage {case_index}/{query_index}"
            );
            assert_eq!(json!(warnings), query["warnings"]);
        }
    }
}
