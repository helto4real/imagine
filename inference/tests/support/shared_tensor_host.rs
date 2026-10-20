use imagine_inference::{
    native::{self, Error},
    tensor_host::{self, HostValue, Scalar},
};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};
pub(super) fn scalar(value: Scalar) -> Value {
    match value {
        Scalar::Bool(value) => json!({"type":"bool","value":value}),
        Scalar::Int(value) => json!({"type":"int","value":value.to_string()}),
        Scalar::UInt(value) => json!({"type":"int","value":value.to_string()}),
        Scalar::Float(value) => json!({"type":"float","bits":format!("{:016x}",value.to_bits())}),
        Scalar::Complex { real, imaginary } => {
            json!({"type":"complex","real":format!("{:016x}",real.to_bits()),"imaginary":format!("{:016x}",imaginary.to_bits())})
        }
    }
}
fn host(value: HostValue) -> Value {
    match value {
        HostValue::Scalar(value) => scalar(value),
        HostValue::List(values) => {
            json!({"type":"list","items":values.into_iter().map(host).collect::<Vec<_>>()})
        }
    }
}
pub(super) fn input(desc: &Value) -> Tensor {
    let shape: Vec<i64> = desc["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    if desc["kind"] == "meta" {
        return super::shared_model_detection_tests::meta(
            &shape,
            if desc["dtype"] == "float32" {
                Kind::Float
            } else {
                Kind::Int64
            },
        );
    }
    if desc["kind"] == "quantized" {
        let n: i64 = shape.iter().product();
        let data = Tensor::arange(n, (Kind::Float, Device::Cpu)).reshape(&shape) * 0.25 + 0.25;
        let kind = match desc["dtype"].as_str().unwrap() {
            "qint8" => Kind::QInt8,
            "quint8" => Kind::QUInt8,
            "qint32" => Kind::QInt32,
            _ => unreachable!(),
        };
        return data.f_quantize_per_tensor(0.125, 0, kind).unwrap();
    }
    let bytes: Vec<u8> = desc["bytes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let raw = unsafe {
        torch_sys::at_tensor_of_blob(
            bytes.as_ptr().cast(),
            shape.as_ptr(),
            shape.len(),
            std::ptr::null(),
            0,
            desc["code"].as_i64().unwrap() as i32,
            -1,
        )
    };
    let error = unsafe { torch_sys::get_and_reset_last_err() };
    assert!(error.is_null(), "own initialized blob constructor");
    assert!(!raw.is_null());
    let borrowed = unsafe { Tensor::from_ptr(raw) };
    let value = native::clone_tensor(&borrowed)
        .unwrap()
        .to_device(super::device(desc["device"].as_str().unwrap()));
    match desc["layout"].as_str().unwrap() {
        "transpose" => value.transpose(0, 1),
        "neg" => value.f_internal_neg_view().unwrap(),
        "conj" => value.f_conj().unwrap(),
        _ => value,
    }
}
fn check(result: Result<Value, Error>, expected: &Value) {
    if expected["error"].is_string() {
        super::binding_tests::reference_error(result.unwrap_err(), expected);
    } else {
        assert_eq!(result.unwrap(), expected["output"]);
    }
}
#[test]
fn original_host_item_tolist_preserve_scalar_categories_unsigned_limits_complex_bits_strides_views_quantized_and_meta_errors()
 {
    let _native = super::kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-tensor-host.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 361);
    for case in fixture["cases"].as_array().unwrap() {
        let value = input(&case["input"]);
        check(tensor_host::item(&value).map(scalar), &case["item"]);
        check(tensor_host::tolist(&value).map(host), &case["tolist"]);
    }
}
