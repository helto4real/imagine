//! Explicit, tiny synthetic CUDA tests; invoke under the shared GPU reservation.
use imagine_inference::{
    checkpoint::PreparedParameters, execution_value::Value, models::moss_image as m,
    native::Kitchen, patching as p,
};
use std::{collections::BTreeMap, path::Path};
use tch::{Device, Kind, Tensor};
// Exercise H3's actual terminal release function without exposing a new API.
pub use imagine_inference::{components, memory, native};
#[path = "../src/h3_product/lifecycle.rs"]
mod h3_lifecycle;

#[test]
fn terminal_h3_release_unpins_without_unloading_reusable_cuda_storage() {
    use imagine_inference::{execution::Runtime, registry::BackendRegistry};
    let native = unsafe {
        Kitchen::load(Path::new(
            &std::env::var("IMAGINE_KITCHEN_LIBRARY").unwrap(),
        ))
        .unwrap()
    };
    let runtime = Runtime::new(native.clone(), BackendRegistry::new(None));
    for budget in [1., 1_000_000.] {
        let pins = components::HostPins::with_limit(native.clone(), 8192.).unwrap();
        let state = PreparedParameters::plain(BTreeMap::from([(
            "unit.weight".into(),
            Value::tensor(Tensor::ones([32, 32], (Kind::BFloat16, Device::Cpu))),
        )]));
        let inventory = serde_json::from_value(serde_json::json!({
            "modules":[{"name":"unit","module_bytes":2048,"offload_bytes":2048,
                "cast_weights":true,"parameters":["unit.weight"]}],
            "model_size":2048,"manual_cast_dtype":"torch.bfloat16","buffers":[]
        }))
        .unwrap();
        let owner = components::static_component_from_inventory(&runtime, state, inventory)
            .unwrap()
            .with_shared_pins(pins.clone());
        let model = components::StaticModel::new(owner, &runtime, Some(Kind::BFloat16));
        let patcher = model.patcher();
        patcher
            .model
            .partially_load(Device::Cuda(0), budget, false)
            .unwrap();
        let before = model.owner.borrow().parameters["unit.weight"]
            .as_tensor()
            .unwrap();
        let residency = model.owner.borrow().placement.loaded_bytes;
        let witness = model.clone();
        h3_lifecycle::release_component(model, patcher, &mut memory::Manager::new()).unwrap();
        let after = witness.owner.borrow().parameters["unit.weight"]
            .as_tensor()
            .unwrap();
        assert_eq!(before.data_ptr(), after.data_ptr());
        assert_eq!(before.device(), after.device());
        assert_eq!(residency, witness.owner.borrow().placement.loaded_bytes);
        assert_eq!(pins.bytes(), 0);
        assert_eq!(witness.owner.borrow().pinned_bytes, 0);
    }
}
#[test]
fn moss_qkv_patches_change_only_selected_rows_and_preserve_bf16_bits() {
    let native = unsafe {
        Kitchen::load(Path::new(
            &std::env::var("IMAGINE_KITCHEN_LIBRARY").unwrap(),
        ))
        .unwrap()
    };
    let _grad = tch::no_grad_guard();
    let map = m::moss_key_map(&[(
        "diffusion_model.layers.0.proj.qkv.weight".into(),
        vec![6, 4],
    )]);
    for (i, projection) in ["q", "k", "v"].iter().enumerate() {
        let base = Tensor::ones([6, 4], (Kind::BFloat16, Device::Cuda(0)));
        let state = PreparedParameters::plain(BTreeMap::from([(
            "layers.0.proj.qkv.weight".into(),
            Value::tensor(base.shallow_clone()),
        )]));
        let prefix = format!("diffusion_model.layers.0.proj.{projection}");
        let source = BTreeMap::from([
            (
                format!("{prefix}.lora_up.weight"),
                Value::tensor(Tensor::ones([2, 2], (Kind::BFloat16, Device::Cpu))),
            ),
            (
                format!("{prefix}.lora_down.weight"),
                Value::tensor(Tensor::ones([2, 4], (Kind::BFloat16, Device::Cpu))),
            ),
        ]);
        let owner = p::bind_component(
            "diffusion_model.",
            &state,
            &[(source, 0.25, 0.25)],
            &map,
            false,
        )
        .unwrap();
        let mut diagnostics = p::Diagnostics::default();
        let output = p::calculate_weight(
            &native,
            &owner.patches["layers.0.proj.qkv.weight"],
            base.copy(),
            "layers.0.proj.qkv.weight",
            Kind::Float,
            None,
            &mut diagnostics,
        )
        .unwrap()
        .to_kind(Kind::BFloat16);
        let mut expected = vec![1f32; 24];
        expected[i * 8..i * 8 + 8].fill(1.5);
        let expected = Tensor::from_slice(&expected)
            .reshape([6, 4])
            .to_kind(Kind::BFloat16);
        assert!(output.to_device(Device::Cpu).equal(&expected));
        assert!(base.equal(&Tensor::ones_like(&base)));
        assert!(diagnostics.errors.is_empty());
    }
}
