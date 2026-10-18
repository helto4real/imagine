use super::binding_tests::{check, restore};
use imagine_inference::{execution_value::Value as V, models::moss_image as m};
use serde_json::Value;
use tch::Device;
#[test]
fn moss_original_flux2_diffusers_checkpoint_key_shape_and_storage_conversion_matches() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-vae-conversion.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let state = case["input_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| {
                let name = name.as_str().unwrap();
                let value = &case["input"][name];
                (
                    name.to_owned(),
                    restore(value, Device::Cpu).as_tensor().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let shapes = m::flux2_vae_checkpoint_shapes(
            state
                .iter()
                .map(|(name, value)| (name.clone(), value.size()))
                .collect(),
        );
        let pointers = state.iter().map(|(_, v)| v.data_ptr()).collect::<Vec<_>>();
        let converted = m::flux2_vae_checkpoint(state).unwrap();
        assert_eq!(converted.len(), case["output"].as_object().unwrap().len());
        for (name, value) in converted {
            check(
                &V::tensor(value.shallow_clone()),
                &case["output"][&name],
                Device::Cpu,
                &name,
            );
            assert_eq!(value.size(), shapes[&name]);
            assert!(
                pointers.contains(&value.data_ptr()),
                "Source reshape must retain storage"
            );
        }
    }
}
