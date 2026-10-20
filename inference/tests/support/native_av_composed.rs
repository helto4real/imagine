use imagine_inference::{
    execution_value::Value as V,
    models::minimax_h3::{conditioning, flow, preprocessing, streams::NestedTensor as N},
    native,
    sampling::{
        model_sampling::{Config, Flavor, ModelSampling},
        parameterization::{Parameterization, Prediction},
        schedules::ScheduleModel,
    },
};
use serde_json::Value;
use tch::{Device, Kind, Tensor};
fn kind(s: &str) -> Kind {
    match s {
        "float32" => Kind::Float,
        "bfloat16" => Kind::BFloat16,
        "float64" => Kind::Double,
        _ => panic!("fixture dtype"),
    }
}
fn check(t: &Tensor, v: &Value) {
    assert_eq!(serde_json::json!(t.size()), v["shape"]);
    assert_eq!(t.kind(), kind(v["dtype"].as_str().unwrap()));
    assert_eq!(super::bits(t), super::exact_hex(&v["bytes"]));
}
fn streams(n: &N, v: &Value) {
    assert_eq!(n.unbind().borrow().len(), v.as_array().unwrap().len());
    for (t, e) in n.unbind().borrow().iter().zip(v.as_array().unwrap()) {
        check(&t.as_tensor().unwrap(), e);
    }
}
#[test]
fn h3_composed_latent_pack_global_noise_original_flow_and_unpack_match_without_learned_weights() {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-composed.json")).unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    let kitchen = super::kitchen();
    let config = Config {
        shift: 12.0,
        audio_shift: Some(3.0),
        ..Config::default()
    };
    let sampling = ModelSampling::new(&kitchen, Flavor::AudioVideo, config, None).unwrap();
    let prediction = Parameterization {
        prediction: Prediction::Const,
        sigma_data: 1.0,
        noise_scale: 1.0,
    };
    for c in cases {
        let k = kind(c["dtype"].as_str().unwrap());
        let frames = c["frames"].as_i64().unwrap();
        let base = Tensor::zeros([1, 32, frames, 2, 2], (k, Device::Cpu));
        let values = preprocessing::empty_streams(&base).unwrap();
        let video = values[0].as_tensor().unwrap();
        let pixels = Tensor::arange(video.numel() as i64, (Kind::Float, Device::Cpu))
            .reshape(video.size())
            .g_div_scalar(127)
            .to_kind(k);
        video.shallow_clone().f_copy_(&pixels).unwrap();
        let original = N::new(values.to_vec());
        streams(&original, &c["original"]);
        let (packed, shapes) = original.pack().unwrap();
        check(&packed.as_tensor().unwrap(), &c["packed"]);
        let shape = video.size();
        let count = if c["shape_metadata"] == true {
            Some(2)
        } else {
            None
        };
        let scale = conditioning::audio_scale(count, sampling.audio_scale().unwrap());
        assert_eq!(Some(scale), c["audio_scale"].as_f64());
        check(&sampling.sigmas, &c["sigmas"]);
        let prepared = flow::process_packed_in(&packed, &shape, scale, &kitchen).unwrap();
        check(&prepared.as_tensor().unwrap(), &c["prepared"]);
        streams(
            &flow::process_streams_in(&original, scale).unwrap(),
            &c["nested_prepared"],
        );
        let noise = original.prepare_noise(42, None).unwrap().pack().unwrap().0;
        check(&noise.as_tensor().unwrap(), &c["noise"]);
        let sigma = sampling.sigma(&Tensor::from_slice(&[500.0_f32])).unwrap();
        check(&sigma, &c["sigma"]);
        let x = prediction
            .noise_scaling(
                &sigma,
                &noise.as_tensor().unwrap(),
                &prepared.as_tensor().unwrap(),
                false,
            )
            .unwrap();
        check(&x, &c["x"]);
        let velocity = x
            .f_mul_scalar(0.125)
            .unwrap()
            .f_add(&sigma.reshape([1, 1, 1]).f_mul_scalar(0.01).unwrap())
            .unwrap();
        let denoised = prediction.denoised(&sigma, &velocity, &x).unwrap();
        check(&denoised, &c["denoised"]);
        let final_value = V::tensor(
            prediction
                .inverse_noise_scaling(&Tensor::from_slice(&[0.0_f32]), &denoised)
                .unwrap(),
        );
        let output = flow::process_packed_out(&final_value, &shape, scale, &kitchen).unwrap();
        check(&output.as_tensor().unwrap(), &c["output"]);
        let restored = N::unpack(&output, &shapes).unwrap();
        streams(&restored, &c["streams"]);
        for value in restored.unbind().borrow().iter() {
            assert_eq!(
                native::storage_info(&value.as_tensor().unwrap()).unwrap().0,
                native::storage_info(&output.as_tensor().unwrap())
                    .unwrap()
                    .0
            );
        }
        streams(
            &flow::process_streams_out(&N::unpack(&final_value, &shapes).unwrap(), scale).unwrap(),
            &c["nested_output"],
        );
    }
}

#[test]
fn h3_zero_audio_scale_preserves_original_process_out_exception_before_any_latent_operation() {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-composed.json")).unwrap();
    for c in f["errors"].as_array().unwrap() {
        let error = if c["nested"] == true {
            flow::process_streams_out(&N::new(vec![]), 0.0)
                .err()
                .unwrap()
        } else {
            flow::process_packed_out(&V::None, &[], 0.0, &super::kitchen())
                .err()
                .unwrap()
        };
        match error {
            native::Error::ReferenceException { class, message } => {
                assert_eq!(class, c["type"].as_str().unwrap());
                assert_eq!(message, c["message"].as_str().unwrap());
            }
            e => panic!("{e:?}"),
        }
    }
}
