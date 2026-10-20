use imagine_inference::{
    execution_value::Value as V, models::minimax_h3::streams::NestedTensor as N,
};
use serde_json::Value;
use std::rc::Rc;
use tch::{Device, Kind, Tensor};
fn kind(s: &str) -> Kind {
    match s {
        "float32" => Kind::Float,
        "bfloat16" => Kind::BFloat16,
        "float64" => Kind::Double,
        "int64" => Kind::Int64,
        "uint8" => Kind::Uint8,
        _ => panic!("fixture dtype"),
    }
}
fn streams(dtype: Kind, count: usize) -> Vec<V> {
    [vec![1, 24, 2, 2, 2], vec![1, 32, 2, 6], vec![1, 3, 2]]
        .iter()
        .take(count)
        .map(|shape| {
            V::tensor(
                Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, Device::Cpu))
                    .reshape(shape)
                    .g_div_scalar(127)
                    .to_kind(dtype),
            )
        })
        .collect()
}
fn check(t: &Tensor, v: &Value) {
    assert_eq!(serde_json::json!(t.size()), v["shape"]);
    assert_eq!(t.kind(), kind(v["dtype"].as_str().unwrap()));
    assert_eq!(super::bits(t), super::exact_hex(&v["bytes"]));
}
fn check_streams(n: &N, v: &Value) {
    for (t, e) in n.unbind().borrow().iter().zip(v.as_array().unwrap()) {
        check(&t.as_tensor().unwrap(), e);
    }
}
#[test]
fn h3_nested_stream_scaling_and_mutable_list_owners_match_original_class() {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-streams.json")).unwrap();
    let cases = f["scales"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    for c in cases {
        let values = streams(
            kind(c["dtype"].as_str().unwrap()),
            c["count"].as_u64().unwrap() as usize,
        );
        let n = N::new(values.clone());
        let list = n.unbind();
        let copy = n.shallow_copy();
        let out = n.scale_audio(c["scale"].as_f64().unwrap()).unwrap();
        assert_eq!(out.same_object(&n), c["same"].as_bool().unwrap());
        assert_eq!(!copy.same_object(&n), c["copy_new"].as_bool().unwrap());
        assert_eq!(
            !Rc::ptr_eq(&copy.unbind(), &list),
            c["copy_list_new"].as_bool().unwrap()
        );
        assert_eq!(
            Rc::ptr_eq(&list, &n.unbind()),
            c["unbind_same"].as_bool().unwrap()
        );
        for (i, v) in values.iter().enumerate() {
            assert_eq!(
                out.unbind().borrow()[i].same_object(v),
                c["same_streams"][i].as_bool().unwrap()
            );
            assert_eq!(
                copy.unbind().borrow()[i].same_object(v),
                c["copy_streams"][i].as_bool().unwrap()
            );
        }
        check_streams(&out, &c["outputs"]);
        list.borrow_mut().push(V::Int(7));
        assert_eq!(n.unbind().borrow().len(), values.len() + 1);
        assert_eq!(copy.unbind().borrow().len(), values.len());
    }
}
#[test]
fn h3_noise_uses_one_shared_original_global_cpu_generator_across_all_streams_and_indices() {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-streams.json")).unwrap();
    let cases = f["noises"].as_array().unwrap();
    assert_eq!(cases.len(), 72);
    for c in cases {
        let n = N::new(streams(kind(c["dtype"].as_str().unwrap()), 3));
        let indices = c["indices"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_i64().unwrap()).collect::<Vec<_>>());
        let out = n.prepare_noise(c["seed"].as_u64().unwrap(), indices.as_deref());
        if c["error"].is_object() {
            match out.err().expect("original error") {
                imagine_inference::native::Error::ReferenceException { class, message } => {
                    assert_eq!(class, c["error"]["type"].as_str().unwrap());
                    assert_eq!(message, c["error"]["message"].as_str().unwrap());
                }
                e => panic!("{e:?}"),
            }
        } else {
            check_streams(&out.unwrap(), &c["outputs"]);
        }
        let state = N::rng_state().unwrap();
        check(&state, &c["rng"]);
        let empty = n.prepare_empty_noise(&super::kitchen()).unwrap();
        check_streams(&empty, &c["empty"]);
        assert_eq!(super::bits(&N::rng_state().unwrap()), super::bits(&state));
        assert_eq!(c["empty_rng_unchanged"], true);
    }
}
