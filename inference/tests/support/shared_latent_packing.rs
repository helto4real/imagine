use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    context_windows::{LatentShape, LatentShapes, pack_latents, unpack_latents},
    execution_value::Value as V,
    native::{self, Error},
};
use serde_json::{Value, json};
use std::rc::Rc;
fn storage(a: &V, b: &V) -> bool {
    native::storage_info(&a.as_tensor().unwrap()).unwrap().0
        == native::storage_info(&b.as_tensor().unwrap()).unwrap().0
}
fn shape_values(shapes: &LatentShapes) -> V {
    V::List(
        shapes
            .entries()
            .unwrap()
            .iter()
            .map(|v| V::Tuple(v.values().unwrap().into_iter().map(V::Int).collect()))
            .collect(),
    )
}
#[test]
fn typed_c10_value_type_not_implemented_buffer_and_syntax_errors_survive_the_native_abi() {
    use std::ffi::{CStr, CString, c_char};
    unsafe extern "C" {
        fn at_imagine_test_error(kind: i32, code: i32, message: *const c_char);
    }
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-error-messages.json")).unwrap();
    assert_eq!(fixture["torch_python_imported"], false);
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 37);
    for (kind, class) in [
        (10, "ValueError"),
        (11, "TypeError"),
        (12, "NotImplementedError"),
        (13, "BufferError"),
        (14, "SyntaxError"),
    ] {
        for case in fixture["cases"].as_array().unwrap() {
            let message = CString::new(case["input"].as_str().unwrap()).unwrap();
            unsafe {
                at_imagine_test_error(kind, 0, message.as_ptr());
            }
            let raw = unsafe { torch_sys::get_and_reset_last_err() };
            assert!(!raw.is_null());
            let text = unsafe { CStr::from_ptr(raw) }.to_str().unwrap().to_owned();
            unsafe {
                libc::free(raw.cast());
            }
            let e: Error = tch::TchError::Torch(text).into();
            match e {
                Error::ReferenceException {
                    class: actual,
                    message,
                } => {
                    assert_eq!(actual, class);
                    assert_eq!(message, case["output"].as_str().unwrap());
                }
                e => panic!("lost typed c10 class: {e:?}"),
            }
            assert!(unsafe { torch_sys::get_and_reset_last_err() }.is_null());
        }
    }
}
#[test]
fn original_multimodal_latent_packing_preserves_shapes_single_modality_shortcut_views_and_errors() {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-latent-packing.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 546);
    for (i, c) in cases.iter().enumerate() {
        let p = &c["params"];
        let dev = device(p["device"].as_str().unwrap());
        let note = format!("latent packing {i} {} {p}", c["name"]);
        let out: Result<V, Error> = if c["name"] == "pack" {
            let V::List(inputs) = restore(&c["inputs"], dev) else {
                panic!("{note}")
            };
            pack_latents(&inputs).and_then(|(latent,shapes)|{
    let values=unpack_latents(&latent,&shapes)?;
    let owners=json!({"shape_types":shapes.entries()?.iter().map(|v|if matches!(v,LatentShape::Size(_)){"Size"}else{"invalid"}).collect::<Vec<_>>(),"packed_input_storage":inputs.iter().map(|v|storage(&latent,v)).collect::<Vec<_>>(),"unpacked_storage":values.iter().map(|v|storage(v,&latent)).collect::<Vec<_>>(),"unpacked_objects":values.iter().map(|v|v.same_object(&latent)).collect::<Vec<_>>()});
    assert_eq!(owners,c["owners"],"{note}");Ok(V::Tuple(vec![latent,shape_values(&shapes),V::List(values)]))
   })
        } else {
            let combined = restore(&c["input"], dev);
            let shapes = LatentShapes::new(
                p["metadata"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|shape| {
                        LatentShape::Tuple(Rc::new(
                            shape
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_i64().unwrap())
                                .collect(),
                        ))
                    })
                    .collect(),
            );
            unpack_latents(&combined,&shapes).map(|values|{
    assert_eq!(json!({"same_object":values.iter().map(|v|v.same_object(&combined)).collect::<Vec<_>>(),"same_storage":values.iter().map(|v|storage(v,&combined)).collect::<Vec<_>>(),"storage_matrix":values.iter().map(|a|values.iter().map(|b|storage(a,b)).collect::<Vec<_>>()).collect::<Vec<_>>()}),c["owners"],"{note}");V::List(values)
   })
        };
        match out {
            Ok(v) => {
                assert!(!c["error"].is_string(), "{note}: expected {c}");
                check(&v, &c["output"], dev, &note);
            }
            Err(e) => {
                assert!(c["error"].is_string(), "{note}: unexpected {e}");
                reference_error(e, c);
            }
        }
    }
}
