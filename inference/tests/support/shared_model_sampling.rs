use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    sampling::{
        model_sampling::{self as m, Config, Flavor},
        schedules::ScheduleModel,
    },
};
use serde_json::Value;
use tch::Tensor;
fn check_result(result: Result<V, Error>, expected: &Value, dev: tch::Device, context: &str) {
    if expected["error"].is_string() {
        reference_error(
            result
                .err()
                .unwrap_or_else(|| panic!("Expected source error {context}")),
            expected,
        );
    } else if expected["type"] == "float" {
        let V::Float(number) = result.unwrap() else {
            panic!("Expected scalar result {context}")
        };
        let bits = u64::from_str_radix(
            &expected["bits"]
                .as_str()
                .unwrap()
                .as_bytes()
                .chunks_exact(2)
                .rev()
                .map(|b| std::str::from_utf8(b).unwrap())
                .collect::<String>(),
            16,
        )
        .unwrap();
        assert_eq!(number.to_bits(), bits, "{context}");
    } else {
        check(
            &result.unwrap_or_else(|e| panic!("{context}: {e}")),
            expected,
            dev,
            context,
        );
    }
}
#[test]
fn shared_model_sampling_buffers_maps_percent_and_original_errors_match() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-model-sampling.json")).unwrap();
    let previous = native.default_dtype().unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let dtype = kind(c["kind"].as_str().unwrap());
        native.set_default_dtype(dtype).unwrap();
        let flavor = match c["name"].as_str().unwrap() {
            "discrete" => Flavor::Discrete,
            "discrete_edm" => Flavor::DiscreteEdm,
            "distilled" => Flavor::Distilled,
            "continuous_edm" => Flavor::ContinuousEdm,
            "continuous_v" => Flavor::ContinuousV,
            "discrete_flow" => Flavor::DiscreteFlow,
            "av" => Flavor::AudioVideo,
            "cascade" => Flavor::Cascade,
            "flux" => Flavor::Flux,
            "cosmos" => Flavor::Cosmos,
            _ => unreachable!(),
        };
        let mut config = Config::default();
        if flavor == Flavor::Flux {
            config.shift = 1.15;
            config.timesteps = 10000;
        }
        let options = c["settings"].as_object().unwrap();
        for (key, value) in options {
            match key.as_str() {
                "beta_schedule" => config.beta_schedule = value.as_str().unwrap().into(),
                "zsnr" => config.zsnr = value.as_bool().unwrap(),
                "timesteps" => config.timesteps = value.as_i64().unwrap(),
                "shift" => config.shift = value.as_f64().unwrap(),
                "audio_shift" => config.audio_shift = Some(value.as_f64().unwrap()),
                "sigma_min" => config.sigma_min = value.as_f64().unwrap(),
                "sigma_max" => config.sigma_max = value.as_f64().unwrap(),
                "sigma_data" => config.sigma_data = value.as_f64().unwrap(),
                _ => unreachable!(),
            }
        }
        let result = m::ModelSampling::new(&native, flavor, config, None);
        if c["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("Expected construction error {c}")),
                c,
            );
            continue;
        }
        let model = result.unwrap_or_else(|e| panic!("{c}: {e}"));
        check(
            &V::tensor(model.sigmas.shallow_clone()),
            &c["buffers"]["sigmas"],
            tch::Device::Cpu,
            &format!("{c} sigmas"),
        );
        if let Some(logs) = &model.log_sigmas {
            check(
                &V::tensor(logs.shallow_clone()),
                &c["buffers"]["log_sigmas"],
                tch::Device::Cpu,
                &format!("{c} logs"),
            );
        }
        if c["num_timesteps"].is_i64() {
            assert_eq!(model.num_timesteps, c["num_timesteps"].as_i64().unwrap());
        }
        if c["skip_steps"].is_i64() {
            assert_eq!(model.skip_steps, c["skip_steps"].as_i64().unwrap());
        }
        if flavor == Flavor::AudioVideo {
            check_result(
                model.audio_scale().map(V::Float),
                &c["audio_scale"],
                tch::Device::Cpu,
                "audio scale",
            );
        }
        for p in c["percent"].as_array().unwrap() {
            check_result(
                model.percent_to_sigma(&native, p["p"].as_f64().unwrap()),
                &p["result"],
                tch::Device::Cpu,
                &format!("{} {} percent {}", c["name"], c["kind"], p["p"]),
            );
        }
        for op in c["ops"].as_array().unwrap() {
            let dev = device(op["device"].as_str().unwrap());
            let values: Vec<f64> = op["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let t = if values.is_empty() {
                Tensor::from(0.125)
            } else {
                Tensor::from_slice(&values)
            }
            .to_kind(dtype)
            .to_device(dev);
            let y = if op["op"] == "sigma" {
                model.sigma(&t)
            } else {
                model.timestep(&t)
            };
            check_result(
                y.map(V::tensor),
                &op["result"],
                dev,
                &format!("{} {} {op}", c["name"], c["kind"]),
            );
        }
    }
    native.set_default_dtype(previous).unwrap();
}

#[test]
fn advanced_lcm_denoised_preserves_distilled_timestep_and_global_dtype() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let previous = native.default_dtype().unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-lcm.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let default = kind(c["default"].as_str().unwrap());
        native.set_default_dtype(default).unwrap();
        let model =
            m::ModelSampling::new(&native, Flavor::Distilled, Config::default(), None).unwrap();
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["kind"].as_str().unwrap());
        let shape = c["shape"].as_array().unwrap();
        let sigma = if shape.is_empty() {
            Tensor::from(0.75)
        } else if shape[0] == 1 {
            Tensor::from_slice(&[0.75f64])
        } else {
            Tensor::from_slice(&[0.75f64, 0.25])
        }
        .to_kind(default)
        .to_device(dev);
        let data = |offset| {
            Tensor::arange(12, (tch::Kind::Float, dev))
                .f_add_scalar(offset)
                .unwrap()
                .f_remainder(29)
                .unwrap()
                .f_mul_scalar(0.03125)
                .unwrap()
                .f_sub_scalar(0.40625)
                .unwrap()
                .reshape([2, 1, 2, 3])
                .to_kind(dtype)
        };
        let result = imagine_inference::sampling::parameterization::lcm_denoised(
            &model,
            &sigma,
            &data(7),
            &data(3),
        )
        .unwrap();
        check(&V::tensor(result), &c["output"], dev, &format!("LCM {c}"));
    }
    native.set_default_dtype(previous).unwrap();
}
