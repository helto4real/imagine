use imagine_inference::{
    checkpoint::rebuild_versions::{self, Arguments, Version},
    native::{self, Error},
    patching::options::Options,
};
use serde_json::{Value, json};
use tch::{Device, Tensor};
fn exact_error(error: &Error, expected: &Value) {
    assert!(
        error.is_reference(expected["error"].as_str().unwrap()),
        "{error:?}: {expected}"
    );
    assert_eq!(
        error.to_string(),
        expected["message"].as_str().unwrap(),
        "{expected}"
    );
}
fn state(tensor: &Tensor, source: &Tensor) -> Value {
    json!({"shape":tensor.size(),"stride":tensor.stride(),"grad":tensor.requires_grad(),"conj":tensor.is_conj(),"neg":tensor.is_neg(),"storage_alias":native::storage_info(tensor).unwrap().0==native::storage_info(source).unwrap().0})
}
fn replay(fixture: &Value, count: usize) {
    // Register the same native CUDA kernels as the actual inference worker.
    let _library = super::kitchen();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), count);
    for case in fixture["cases"].as_array().unwrap() {
        let source = Tensor::from_slice(&(0..64).map(|i| (i % 2) as u8).collect::<Vec<_>>())
            .to_device(super::device(case["device"].as_str().unwrap()));
        let hooks = Options::dictionary(vec![]);
        let metadata = case["metadata"]
            .as_object()
            .map(|values| {
                values
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_bool().unwrap()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let operation = || {
            rebuild_versions::dense(
                if case["version"] == 2 {
                    Version::V2
                } else {
                    Version::V3
                },
                Arguments {
                    storage: &source,
                    dtype: case["code"].as_i64().unwrap() as i32,
                    offset: 0,
                    size: &[2],
                    stride: &[1],
                    requires_grad: case["requires_grad"].as_bool().unwrap(),
                    backward_hooks: hooks.clone(),
                    metadata: &metadata,
                },
            )
        };
        let result = if case["no_grad"] == true {
            tch::no_grad(operation)
        } else {
            operation()
        };
        if case["error"].is_string() {
            let error = result.err().expect("original reconstruction failure");
            exact_error(&error, case);
            let error = error.into_exception();
            let frames = error.frames();
            assert_eq!(frames.len(), 1);
            let frame = &frames[0];
            let name = if case["version"] == 2 { "tensor" } else { "t" };
            let partial = frame.locals().get(name);
            assert_eq!(partial.is_some(), case["partial"].is_object(), "{case}");
            if let Some(partial) = partial {
                let partial = partial.downcast::<Tensor>().ok().unwrap();
                for (key, value) in state(&partial, &source).as_object().unwrap() {
                    assert_eq!(value, &case["partial"][key], "partial {key}: {case}");
                }
            }
            let owner = frame
                .locals()
                .get("backward_hooks")
                .unwrap()
                .downcast::<Options>()
                .ok()
                .unwrap();
            assert!(owner.same_object(&hooks));
            continue;
        }
        let output = result.unwrap_or_else(|error| panic!("{case}: {error:?}"));
        assert!(output.backward_hooks.same_object(&hooks));
        for (key, value) in state(&output.tensor, &source).as_object().unwrap() {
            assert_eq!(value, &case[key], "{key}: {case}");
        }
        let observed = (|| -> Result<Tensor, Error> {
            Ok(output
                .tensor
                .f_detach()?
                .f_resolve_conj()?
                .f_resolve_neg()?
                .f_contiguous()?
                .f_view_dtype(tch::Kind::Uint8)?
                .f_to_device(Device::Cpu)?)
        })();
        if case["observation_error"].is_object() {
            exact_error(
                &observed.expect_err("original byte-observation failure"),
                &case["observation_error"],
            );
        } else {
            assert_eq!(
                super::bits(&observed.unwrap()),
                super::exact_hex(&case["payload_hex"]),
                "{case}"
            );
        }
    }
}
#[test]
fn original_v2_v3_tensor_construction_preserves_gradient_order_native_errors_storage_aliases_and_defined_bytes()
 {
    replay(
        &serde_json::from_str(include_str!(
            "../../fixtures/shared-rebuild-constructors.json"
        ))
        .unwrap(),
        64,
    );
}
#[test]
fn original_v2_v3_dense_metadata_preserves_flag_presence_partial_tensor_mutations_hooks_and_exact_default_diagnostics()
 {
    replay(
        &serde_json::from_str(include_str!("../../fixtures/shared-rebuild-versions.json")).unwrap(),
        384,
    );
}
