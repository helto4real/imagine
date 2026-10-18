use super::{binding_tests::check, kind, kitchen};
use imagine_inference::{
    execution_value::Value as V,
    models::{
        moss_image as m,
        qwen_image21::{ImageVaeMemory, VaeOperation},
    },
};
use serde_json::Value;
use tch::{Device, Kind};
#[test]
fn moss_original_discrete_flow_schedules_percent_mapping_and_model_vae_memory_estimates_match() {
    let native = kitchen();
    let _dtype = super::triton_default_dtype_tests::DefaultDtype::preserve(native.clone());
    native.set_default_dtype(Kind::Float).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-sampling.json")).unwrap();
    let mut schedules = 0;
    for group in fixture["schedules"].as_array().unwrap() {
        let model = m::MossSampling::new(
            group["shift"].as_f64().unwrap(),
            group["multiplier"].as_f64().unwrap(),
            1000,
        )
        .unwrap();
        check(
            &V::tensor(model.sigmas.shallow_clone()),
            &group["sigmas"],
            Device::Cpu,
            "discrete-flow buffer",
        );
        for call in group["cases"].as_array().unwrap() {
            let out = model
                .normal(
                    call["steps"].as_i64().unwrap(),
                    call["sgm"].as_bool().unwrap(),
                    &native,
                )
                .unwrap();
            check(
                &V::tensor(out),
                &call["output"],
                Device::Cpu,
                "normal/sgm_uniform",
            );
            schedules += 1;
        }
        for pair in group["percents"].as_array().unwrap() {
            assert_eq!(
                model.percent_to_sigma(pair[0].as_f64().unwrap()).to_bits(),
                pair[1].as_f64().unwrap().to_bits()
            );
        }
    }
    for call in fixture["model_memory"].as_array().unwrap() {
        let shape = call["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let bytes = m::moss_memory_required(
            &shape,
            kind(call["dtype"].as_str().unwrap()),
            call["flash"].as_bool().unwrap(),
        );
        assert_eq!(bytes.to_bits(), call["bytes"].as_f64().unwrap().to_bits());
    }
    for call in fixture["vae_memory"].as_array().unwrap() {
        let shape = call["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let memory = ImageVaeMemory::Flux2 {
            amd: call["amd"].as_bool().unwrap(),
        };
        let dtype = kind(call["dtype"].as_str().unwrap());
        assert_eq!(
            memory
                .required(VaeOperation::Encode, &shape, dtype)
                .to_bits(),
            call["encode"].as_f64().unwrap().to_bits()
        );
        assert_eq!(
            memory
                .required(VaeOperation::Decode, &shape, dtype)
                .to_bits(),
            call["decode"].as_f64().unwrap().to_bits()
        );
    }
    assert_eq!(schedules, 48);
}
