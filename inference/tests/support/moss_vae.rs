use super::{
    binding_tests::{check, restore},
    device, kind, kitchen,
    qwen_text_encoder_tests::data,
};
use imagine_inference::{
    checkpoint::PreparedParameters,
    execution::Runtime,
    execution_value::Value as V,
    models::{moss_image as m, qwen_image21 as q},
    native::Error,
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0)
    }
}
struct Exec {
    operations: usize,
}
impl q::VaeExecution for Exec {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        Ok(())
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor, Error> {
        q::vae_pytorch_attention(rt, q, k, v)
    }
}
#[test]
fn moss_original_flux2_vae_encode_decode_bn_packing_persistent_buffer_dtypes_and_manual_casts_match()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-vae.json")).unwrap();
    // 446ef4e predates the fused path.
    assert_eq!(run(&fixture, false), 18);
}
#[test]
fn moss_fused_flux2_vae_strips_halving_and_level_buffer_match_426f5eea() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/reference-20261027/moss-vae-fused.json"
    ))
    .unwrap();
    assert_eq!(run(&fixture, true), 12);
}
fn run(fixture: &Value, fused: bool) -> usize {
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let native = kitchen();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native, compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let mut count = 0;
    for group in fixture["groups"].as_array().unwrap() {
        let dev = device(group["device"].as_str().unwrap());
        let dtype = kind(group["dtype"].as_str().unwrap());
        let mut state = std::collections::BTreeMap::new();
        for (name, f) in group["factories"].as_object().unwrap() {
            let shape = f["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect::<Vec<_>>();
            let mut x = data(
                &shape,
                Device::Cpu,
                Kind::Float,
                f["offset"].as_i64().unwrap(),
            )
            .f_mul_scalar(0.0625)
            .unwrap();
            if f["norm"] == true {
                x = x.f_add_scalar(1.).unwrap()
            }
            state.insert(name.clone(), x.f_to_kind(dtype).unwrap());
        }
        for (name, value) in group["buffers"].as_object().unwrap() {
            state.insert(
                name.clone(),
                restore(value, dev)
                    .as_tensor()
                    .unwrap()
                    .f_to_device(Device::Cpu)
                    .unwrap(),
            );
        }
        let spec: q::ComponentSchema = serde_json::from_value(group["schema"].clone()).unwrap();
        let p: PreparedParameters = q::prepare_resident_component_with_options(
            &rt,
            "vae",
            &spec,
            state,
            dev,
            q::ComponentLoad {
                dtype,
                freeze: false,
            },
        )
        .unwrap();
        assert_eq!(
            p["bn.num_batches_tracked"].as_tensor().unwrap().kind(),
            Kind::Int64
        );
        for parameter in &spec.parameters {
            assert_eq!(
                p[&parameter.name].as_tensor().unwrap().requires_grad(),
                parameter.requires_grad.unwrap_or(true),
                "{}/constructor requires_grad",
                parameter.name
            );
        }
        for (name, value) in &p.values {
            // Module.state_dict serializes detached tensors; constructor flags
            // are checked above against the original named-parameter schema.
            check(
                &V::tensor(value.as_tensor().unwrap().detach()),
                &group["state"][name],
                dev,
                name,
            );
        }
        let vae = m::Flux2Vae::new(
            m::Flux2VaeConfig {
                dim: 32,
                decoder_dim: 32,
                fused_norm: fused,
                strip_elements: group["strip_elements"].as_i64().unwrap_or(1 << 24),
                ..m::Flux2VaeConfig::default()
            },
            p,
        );
        for (i, call) in group["calls"].as_array().unwrap().iter().enumerate() {
            let x = restore(&call["args"][0], dev).as_tensor().unwrap();
            let mut e = Exec { operations: 0 };
            let out = if call["operation"] == "encode" {
                vae.encode(&rt, &x, &mut e)
            } else {
                vae.decode(&rt, &x, &mut e)
            }
            .unwrap_or_else(|err| panic!("{dev:?}/{dtype:?}/{i}: {err}"));
            check(
                &V::tensor(out),
                &call["output"],
                dev,
                &format!("{dev:?}/{dtype:?}/{i}"),
            );
            if !call["operations"].is_null() {
                assert_eq!(
                    json!(e.operations),
                    call["operations"],
                    "{dev:?}/{dtype:?}/{i}/interrupt-boundaries"
                );
            }
            count += 1;
        }
    }
    count
}
