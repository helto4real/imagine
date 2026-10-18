use super::*;
use imagine_inference::{
    awq_layout::{self as layout, Backend, Dispatch, Operand},
    convrot, layouts,
    native::Error,
    storage::QuantizedStorage,
};

fn integers(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}
fn tensor(v: &Value, dev: Device) -> Tensor {
    let shape = integers(&v["shape"]);
    let strides = integers(&v["strides"]);
    let kind = kind(v["dtype"].as_str().unwrap());
    let data = Tensor::from_slice(&exact_hex(&v["payload_hex"]))
        .view_dtype(kind)
        .reshape(shape.as_slice())
        .to_device(dev);
    if data.stride() == strides {
        data
    } else {
        let mut result = Tensor::empty_strided(shape.as_slice(), strides.as_slice(), (kind, dev));
        result.copy_(&data);
        result
    }
}
fn quantized(v: &Value, dev: Device) -> QuantizedStorage {
    layout::parts(
        tensor(&v["payload"], dev),
        tensor(&v["scale"], dev),
        tensor(&v["zeros"], dev),
        kind(v["orig_dtype"].as_str().unwrap()),
        integers(&v["orig_shape"]),
        v["group"].as_i64().unwrap(),
        v["transposed"].as_bool().unwrap(),
    )
}
fn check_tensor(t: &Tensor, v: &Value, note: &str) {
    assert!(
        bits(t) == exact_hex(&v["payload_hex"]),
        "{note}: exact bytes"
    );
    assert_eq!(serde_json::json!(t.size()), v["shape"], "{note}: shape");
    assert_eq!(
        serde_json::json!(t.stride()),
        v["strides"],
        "{note}: strides"
    );
    assert_eq!(
        t.kind(),
        kind(v["dtype"].as_str().unwrap()),
        "{note}: dtype"
    );
}
fn check_storage(q: &QuantizedStorage, root: &QuantizedStorage, v: &Value) {
    assert_eq!(q.layout, v["value"]["layout"].as_str().unwrap());
    assert_eq!(
        q.params.original_kind,
        kind(v["value"]["orig_dtype"].as_str().unwrap())
    );
    assert_eq!(
        serde_json::json!(q.params.original_shape),
        v["value"]["orig_shape"]
    );
    assert_eq!(
        serde_json::json!(q.params.field_order),
        v["declaration_order"]
    );
    assert_eq!(
        serde_json::json!(q.params.tensors.keys().collect::<Vec<_>>()),
        v["fields"]
    );
    assert_eq!(
        q.params.fields["group_size"],
        imagine_inference::storage::ScalarField::Integer(v["value"]["group"].as_i64().unwrap())
    );
    assert_eq!(
        q.params.fields["transposed"],
        imagine_inference::storage::ScalarField::Bool(v["value"]["transposed"].as_bool().unwrap())
    );
    let state = layout::state_dict(q, "weight").unwrap();
    assert_eq!(
        serde_json::json!(state.iter().map(|(k, _)| k).collect::<Vec<_>>()),
        v["state_keys"]
    );
    let own = layout::plain_tensors(q).unwrap();
    let original = layout::plain_tensors(root).unwrap();
    for (index, ((name, t), (own, original))) in state
        .iter()
        .zip(
            [own.0, own.1, own.2]
                .into_iter()
                .zip([original.0, original.1, original.2]),
        )
        .enumerate()
    {
        check_tensor(t, &v["state"][name]["tensor"], name);
        assert_eq!(
            t.data_ptr() == original.data_ptr(),
            v["state"][name]["root_alias"].as_bool().unwrap()
        );
        assert_eq!(
            t.data_ptr() == own.data_ptr(),
            v["state"][name]["own_alias"].as_bool().unwrap()
        );
        assert_eq!(
            t.data_ptr() == own.data_ptr(),
            v["plain_alias"][index].as_bool().unwrap()
        );
    }
}

#[test]
fn awq_layout_storage_aliases_transpose_export_clone_and_detach_match_reference() {
    let native = kitchen();
    for case in fixtures()["awq_layout_storage"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let root = quantized(&case["input"], dev);
        assert_eq!(serde_json::json!(layout::MIN_SM_VERSION), case["min_sm"]);
        assert_eq!(
            serde_json::json!(layout::QUANTIZES_INPUT),
            case["quantizes_input"]
        );
        check_storage(&root, &root, &case["states"]["original"]);
        let transposed = layout::transpose(&root).unwrap();
        let variants = [
            ("tt", layout::transpose(&transposed).unwrap()),
            ("t", transposed),
            ("clone", root.clone_storage(&native).unwrap()),
            ("detach", root.detach(&native).unwrap()),
        ];
        for (name, q) in variants {
            check_storage(&q, &root, &case["states"][name]);
        }
    }
}

#[test]
fn awq_layout_dequantization_dtype_order_cropping_and_exact_errors_match_reference() {
    let _native = kitchen();
    for (index, case) in fixtures()["awq_layout_decode"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = device(case["device"].as_str().unwrap());
        let q = quantized(&case["input"], dev);
        let result = match case["operation"].as_str().unwrap() {
            "raw" => layout::dequantize_raw(&q.payload, &q.params),
            "quantize" => match layout::quantize(&q.payload) {
                Err(e) => Err(e),
                Ok(_) => panic!("Source quantize rejects all inputs"),
            },
            "transpose_bad_shape" => match layout::transpose(&q) {
                Err(e) => Err(e),
                Ok(_) => panic!("Source rejects short original shape"),
            },
            _ => layout::dequantize(&q),
        };
        if case["error"].is_string() {
            assert_reference_error(result.expect_err("Original rejects this case"), case);
        } else {
            let out = result.unwrap_or_else(|e| panic!("AWQ layout decode {index}: {e}"));
            check_tensor(&out, &case["output"], &format!("AWQ layout decode {index}"));
            assert_eq!(out.device(), dev);
        }
    }
}

enum Owned {
    Plain(Tensor),
    Quantized(QuantizedStorage),
}
impl Owned {
    fn restore(v: &Value, dev: Device) -> Self {
        match v["type"].as_str().unwrap() {
            "plain" => Self::Plain(tensor(&v["tensor"], dev)),
            "awq" => Self::Quantized(quantized(&v["tensor"], dev)),
            "int8" => Self::Quantized(layouts::int8_parts(
                tensor(&v["payload"], dev),
                tensor(&v["scale"], dev),
                kind(v["orig_dtype"].as_str().unwrap()),
                integers(&v["orig_shape"]),
                false,
                false,
                64,
            )),
            _ => unreachable!(),
        }
    }
    fn operand(&self) -> Operand<'_> {
        match self {
            Self::Plain(t) => Operand::Plain(t),
            Self::Quantized(q) => Operand::Quantized(q),
        }
    }
}

#[test]
fn awq_layout_linear_mm_addmm_mixed_operands_bias_and_fallback_match_reference() {
    let native = kitchen();
    for (index, case) in fixtures()["awq_layout_dispatch"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = device(case["device"].as_str().unwrap());
        let backend = if case["backend"] == "eager" {
            Backend::Eager
        } else {
            Backend::Cuda
        };
        let input = Owned::restore(&case["input"], dev);
        let weight = Owned::restore(&case["weight"], dev);
        let bias = if case["bias"].is_null() {
            None
        } else {
            Some(Owned::restore(&case["bias"], dev))
        };
        if backend == Backend::Cuda
            && let Owned::Quantized(q) = &weight
            && case["scenario"] != "transposed_linear"
        {
            // Source-proven capacities for this matrix, never a public gate.
            assert!(q.payload.is_contiguous());
            assert!(q.params.tensors["scale"].is_contiguous());
            assert!(q.params.tensors["zeros"].is_contiguous());
            assert_eq!(
                q.params.fields["group_size"],
                imagine_inference::storage::ScalarField::Integer(64)
            );
        }
        let mut trace = Vec::<String>::new();
        let mut observe = |event: &str| trace.push(event.to_owned());
        let mut dispatch = Dispatch {
            implementation: None,
            native: &native,
            backend,
            observer: Some(&mut observe),
            dequantize_other: |q: &QuantizedStorage| {
                assert_eq!(q.layout, "TensorWiseINT8Layout");
                layouts::dequantize_with(q, |payload, p| {
                    let scale = &p.tensors["scale"];
                    let out = if backend == Backend::Cuda {
                        convrot::dequantize_simple(&native, payload, scale, p.original_kind)?
                    } else {
                        eager::dequantize_int8(payload, scale)?.f_to_kind(p.original_kind)?
                    };
                    Ok::<_, Error>(out.f_to_kind(p.original_kind)?)
                })
            },
        };
        // Handler kwargs are intentionally not forwarded by the frozen code,
        // including alpha/beta. Fixture options test that source behavior.
        let result = unsafe {
            match case["operation"].as_str().unwrap() {
                "linear" => dispatch.linear(
                    input.operand(),
                    weight.operand(),
                    bias.as_ref().map(Owned::operand),
                ),
                "mm" => dispatch.mm(input.operand(), weight.operand()),
                "addmm" => dispatch.addmm(
                    bias.as_ref().unwrap().operand(),
                    input.operand(),
                    weight.operand(),
                ),
                _ => unreachable!(),
            }
        };
        if case["error"].is_string() {
            assert_reference_error(result.unwrap_err(), case);
        } else {
            let out = result.unwrap_or_else(|e| {
                panic!(
                    "AWQ dispatch {index}, {} {}: {e}",
                    case["operation"], case["scenario"]
                )
            });
            check_tensor(
                &out,
                &case["output"],
                &format!(
                    "AWQ dispatch {index}, {} {}",
                    case["operation"], case["scenario"]
                ),
            );
            assert_eq!(out.device(), dev);
        }
        assert_eq!(
            serde_json::json!(trace),
            case["trace"],
            "AWQ dispatch {index}: original evaluation order"
        );
    }
}
