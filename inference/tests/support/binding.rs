use super::*;
use imagine_inference::{
    execution::{Callable, Runtime},
    execution_value::{Kwargs, Value as V},
    native::Error,
    registry::{BackendRegistry, FunctionConstraints},
    storage::{LayoutParams, QuantizedStorage, ScalarField},
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn fields(v: &Value) -> ScalarField {
    if v.is_null() {
        ScalarField::None
    } else if let Some(b) = v.as_bool() {
        ScalarField::Bool(b)
    } else if let Some(n) = v.as_i64() {
        ScalarField::Integer(n)
    } else if let Some(n) = v.as_f64() {
        ScalarField::Float(n)
    } else if let Some(s) = v.as_str() {
        ScalarField::Text(s.into())
    } else {
        ScalarField::Shape(ints(v))
    }
}
pub(super) fn restore(v: &Value, dev: Device) -> V {
    match v["type"].as_str().unwrap() {
        "tensor_expression" => {
            if v["operation"] == "efficient_zero" {
                return V::tensor(
                    Tensor::f_internal_efficientzerotensor(
                        ints(&v["shape"]).as_slice(),
                        (
                            kind(v["dtype"].as_str().unwrap()),
                            device(v["device"].as_str().unwrap()),
                        ),
                    )
                    .unwrap(),
                );
            }
            let mut source = restore(&v["source"], dev).as_tensor().unwrap();
            V::tensor(match v["operation"].as_str().unwrap() {
                "conj" => source.f_conj().unwrap(),
                "neg_view" => source.f_internal_neg_view().unwrap(),
                "invalid_grad_view" => {
                    // The original factory creates a leaf without a view base.
                    // Deserialized bytes were reshaped; detach that decode view.
                    source = source.f_detach().unwrap().set_requires_grad(true);
                    let _guard = tch::no_grad_guard();
                    let view = source.f_view(ints(&v["shape"]).as_slice()).unwrap();
                    let _ = source.f_add_scalar_(1).unwrap();
                    view
                }
                "view" => source.f_view(ints(&v["shape"]).as_slice()).unwrap(),
                "mul" => source
                    .f_mul(&restore(&v["other"], dev).as_tensor().unwrap())
                    .unwrap(),
                _ => panic!("Unknown owned fixture tensor expression"),
            })
        }
        "tensor" => {
            let dev = v["device"].as_str().map(device).unwrap_or(dev);
            let shape = ints(&v["shape"]);
            let strides = ints(&v["strides"]);
            if v["storage"].is_object() {
                let base = restore(&v["storage"], dev).as_tensor().unwrap();
                let t = base.as_strided(
                    shape.as_slice(),
                    strides.as_slice(),
                    v["storage_offset"].as_i64(),
                );
                return V::tensor(if v["requires_grad"].as_bool().unwrap_or(false) {
                    t.set_requires_grad(true)
                } else {
                    t
                });
            }
            let name = v["dtype"].as_str().unwrap();
            let shell = name == "torch.float8_e8m0fnu";
            let d = if shell { Kind::Uint8 } else { kind(name) };
            let raw = Tensor::from_slice(&exact_hex(&v["payload_hex"]))
                .view_dtype(d)
                .to_device(dev);
            let raw = if shell {
                kitchen()
                    .view_shell_dtype(&raw, imagine_inference::native::ShellDType::E8M0)
                    .unwrap()
            } else {
                raw
            };
            let t = raw.reshape(shape.as_slice()).to_device(dev);
            let t = if t.stride() == strides {
                t
            } else {
                let mut out = Tensor::empty_strided(shape.as_slice(), strides.as_slice(), (d, dev));
                out.copy_(&t);
                out
            };
            V::tensor(if v["requires_grad"].as_bool().unwrap_or(false) {
                t.set_requires_grad(true)
            } else {
                t
            })
        }
        "quantized" => V::quantized(QuantizedStorage::from_validated_parts(
            restore(&v["payload"], dev).as_tensor().unwrap(),
            v["layout"].as_str().unwrap().into(),
            LayoutParams {
                original_kind: kind(v["orig_dtype"].as_str().unwrap()),
                original_shape: ints(&v["orig_shape"]),
                tensors: v["tensors"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), restore(v, dev).as_tensor().unwrap()))
                    .collect(),
                fields: v["fields"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), fields(v)))
                    .collect(),
                field_order: v["field_order"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().into())
                    .collect(),
            },
        )),
        "dtype" => V::DType(kind(v["name"].as_str().unwrap())),
        "device" => V::Device(device(v["name"].as_str().unwrap())),
        "tuple" => V::Tuple(
            v["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, dev))
                .collect(),
        ),
        "list" => V::List(
            v["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, dev))
                .collect(),
        ),
        "dict" => V::Dict(
            v["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_str().unwrap().into(), restore(&p[1], dev)))
                .collect(),
        ),
        "scalar" => {
            if let Some(bits) = v["float_bits"].as_str() {
                return V::Float(f64::from_bits(u64::from_str_radix(bits, 16).unwrap()));
            }
            let v = &v["value"];
            if v.is_null() {
                V::None
            } else if let Some(b) = v.as_bool() {
                V::Bool(b)
            } else if let Some(n) = v.as_i64() {
                V::Int(n)
            } else if let Some(n) = v.as_u64() {
                V::UInt(n)
            } else if let Some(n) = v.as_f64() {
                V::Float(n)
            } else {
                V::Text(v.as_str().unwrap().into())
            }
        }
        _ => panic!("Unknown binding fixture"),
    }
}
pub(super) fn check(value: &V, expected: &Value, dev: Device, note: &str) {
    match value {
        V::Tensor(_) => {
            let t = value.as_tensor().unwrap();
            if let Some(expected_hash) = expected["payload_sha256"].as_str() {
                use sha2::{Digest, Sha256};
                assert_eq!(
                    format!("{:x}", Sha256::digest(bits(&t))),
                    expected_hash,
                    "{note}: exact output bytes SHA256"
                );
            } else {
                assert!(
                    bits(&t) == exact_hex(&expected["payload_hex"]),
                    "{note}: exact output bytes"
                );
            }
            assert_eq!(serde_json::json!(t.size()), expected["shape"], "{note}");
            assert_eq!(serde_json::json!(t.stride()), expected["strides"], "{note}");
            if expected["dtype"] == "torch.float8_e8m0fnu" {
                assert_eq!(kitchen().tensor_type(&t).unwrap().scalar_type, 44);
            } else {
                assert_eq!(t.kind(), kind(expected["dtype"].as_str().unwrap()));
            }
            assert_eq!(
                t.device(),
                expected["device"].as_str().map(device).unwrap_or(dev)
            );
            if let Some(grad) = expected["requires_grad"].as_bool() {
                assert_eq!(t.requires_grad(), grad, "{note}: requires_grad");
            }
        }
        V::Quantized(_) => {
            let q = value.as_quantized().unwrap();
            assert_eq!(q.layout, expected["layout"].as_str().unwrap(), "{note}");
            assert_eq!(
                q.params.original_kind,
                kind(expected["orig_dtype"].as_str().unwrap())
            );
            assert_eq!(
                serde_json::json!(q.params.original_shape),
                expected["orig_shape"]
            );
            assert_eq!(
                serde_json::json!(q.params.field_order),
                expected["field_order"]
            );
            check(
                &V::tensor(q.payload.shallow_clone()),
                &expected["payload"],
                dev,
                note,
            );
            for (k, t) in &q.params.tensors {
                check(
                    &V::tensor(t.shallow_clone()),
                    &expected["tensors"][k],
                    dev,
                    note,
                );
            }
            for (k, v) in &q.params.fields {
                assert_eq!(*v, fields(&expected["fields"][k]), "{note}:{k}");
            }
        }
        V::Tuple(v) | V::List(v) => {
            assert_eq!(
                expected["type"],
                if matches!(value, V::Tuple(_)) {
                    "tuple"
                } else {
                    "list"
                }
            );
            assert_eq!(v.len(), expected["items"].as_array().unwrap().len());
            for (v, e) in v.iter().zip(expected["items"].as_array().unwrap()) {
                check(v, e, dev, note);
            }
        }
        V::NamedTuple(v) => {
            assert_eq!(expected["type"], "namedtuple");
            assert_eq!(expected["name"], v.class.name);
            let items = expected["items"].as_array().unwrap();
            assert_eq!(items.len(), v.fields.len());
            for ((name, value), item) in v.fields.iter().zip(items) {
                assert_eq!(*name, item[0].as_str().unwrap());
                check(value, &item[1], dev, note);
            }
        }
        V::Dict(v) => {
            let items = expected["items"].as_array().unwrap();
            assert_eq!(v.len(), items.len());
            for ((k, v), e) in v.iter().zip(items) {
                assert_eq!(k, e[0].as_str().unwrap());
                check(v, &e[1], dev, note);
            }
        }
        V::None => assert!(expected["value"].is_null()),
        V::Bool(b) => assert_eq!(*b, expected["value"].as_bool().unwrap()),
        V::UInt(n) => assert_eq!(serde_json::json!(n), expected["value"], "{note}"),
        V::Int(n) => assert_eq!(*n, expected["value"].as_i64().unwrap()),
        V::Float(n) => {
            if let Some(bits) = expected["float_bits"].as_str() {
                assert_eq!(
                    n.to_bits(),
                    u64::from_str_radix(bits, 16).unwrap(),
                    "{note}"
                );
            } else {
                assert_eq!(*n, expected["value"].as_f64().unwrap(), "{note}");
            }
        }
        V::Text(s) => assert_eq!(s, expected["value"].as_str().unwrap()),
        V::DType(k) => assert_eq!(*k, kind(expected["name"].as_str().unwrap())),
        V::Device(d) => assert_eq!(*d, device(expected["name"].as_str().unwrap())),
    }
}
pub(super) fn kwargs(v: &Value, dev: Device) -> Kwargs {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_str().unwrap().into(), restore(&p[1], dev)))
        .collect()
}
/// Set literals with their members sorted, as the backend contract renders them:
/// CPython iterates a set of dtypes in an order that follows the objects'
/// addresses in that process, not the source.
fn sorted_sets(message: &str) -> String {
    let mut out = String::new();
    let mut rest = message;
    while let Some(start) = rest.find('{') {
        let Some(len) = rest[start..].find('}') else {
            break;
        };
        out.push_str(&rest[..start]);
        let mut members = rest[start + 1..start + len].split(", ").collect::<Vec<_>>();
        members.sort_unstable();
        out.push('{');
        out.push_str(&members.join(", "));
        out.push('}');
        rest = &rest[start + len + 1..];
    }
    out.push_str(rest);
    out
}
pub(super) fn reference_error(e: Error, c: &Value) {
    match e {
        Error::Selection(e) => {
            assert_eq!(e.exception_class(), c["error"].as_str().unwrap());
            assert_eq!(
                sorted_sets(&e.to_string()),
                sorted_sets(c["message"].as_str().unwrap())
            );
        }
        e => assert_reference_error(e, c),
    }
}

#[test]
fn ordinary_aten_preserves_original_object_identity_separately_from_storage_aliases() {
    fn leaves(v: &V) -> Vec<V> {
        match v {
            V::Tensor(_) => vec![v.clone()],
            V::Tuple(v) | V::List(v) => v.iter().flat_map(leaves).collect(),
            _ => vec![],
        }
    }
    fn restore_identity(v: &mut V, indices: &[Value], inputs: &mut Vec<V>) {
        match v {
            V::Tensor(_) => {
                let index = indices[inputs.len()].as_u64().unwrap() as usize;
                if index < inputs.len() {
                    *v = inputs[index].clone();
                }
                inputs.push(v.clone());
            }
            V::Tuple(v) | V::List(v) => {
                for v in v {
                    restore_identity(v, indices, inputs);
                }
            }
            _ => {}
        }
    }
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    for c in fixtures()["aten_object_identity"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let mut args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, dev))
            .collect::<Vec<_>>();
        let mut inputs = vec![];
        for v in &mut args {
            restore_identity(
                v,
                c["input_object_indices"].as_array().unwrap(),
                &mut inputs,
            );
        }
        let out = unsafe {
            rt.op(c["operation"].as_str().unwrap(), &args, &vec![])
                .unwrap()
        };
        check(&out, &c["output"], dev, "Original ordinary ATen identity");
        let outputs = leaves(&out);
        assert_eq!(
            serde_json::json!(
                outputs
                    .iter()
                    .map(|o| inputs.iter().map(|i| o.same_object(i)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            c["input_identities"]
        );
        assert_eq!(
            serde_json::json!(
                outputs
                    .iter()
                    .map(|o| inputs
                        .iter()
                        .map(|i| o.as_tensor().unwrap().data_ptr()
                            == i.as_tensor().unwrap().data_ptr())
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            c["input_aliases"]
        );
        assert_eq!(
            serde_json::json!(
                outputs
                    .iter()
                    .map(|o| outputs.iter().map(|p| o.same_object(p)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            c["output_identities"]
        );
    }
}

#[test]
fn binding_general_operators_reach_original_backend_and_exact_aten_fallback() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let trace = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let observed = trace.clone();
    rt.backends.observe_selection(Some(Arc::new(move |op, b| {
        observed.lock().unwrap().push(vec![op.into(), b.into()])
    })));
    for (i, c) in fixtures()["binding_operations"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        trace.lock().unwrap().clear();
        let dev = device(c["device"].as_str().unwrap());
        let args: Vec<_> = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, dev))
            .collect();
        let kw = kwargs(&c["kwargs"], dev);
        let guard = c["override"]
            .as_str()
            .map(|s| rt.backends.policy.use_backend(s).unwrap());
        let out = unsafe { rt.dispatch(c["operation"].as_str().unwrap(), &args, &kw) };
        if c["error"].is_string() {
            reference_error(
                out.err()
                    .unwrap_or_else(|| panic!("Expected source rejection at {i}")),
                c,
            );
        } else {
            let value =
                out.unwrap_or_else(|e| panic!("Binding operation {i} {}: {e}", c["operation"]));
            check(
                &value,
                &c["output"],
                dev,
                &format!("binding {i} {}", c["operation"]),
            );
            assert_eq!(
                args.first().is_some_and(|a| a.same_object(&value)),
                c["same_object"].as_bool().unwrap()
            );
            if let Some(alias) = c["payload_alias"].as_bool() {
                assert_eq!(
                    value.as_quantized().unwrap().payload.data_ptr()
                        == args[0].as_quantized().unwrap().payload.data_ptr(),
                    alias
                );
            }
        }
        assert_eq!(
            serde_json::json!(*trace.lock().unwrap()),
            c["backend_calls"],
            "backend trace {i}"
        );
        drop(guard);
    }
}
#[test]
fn binding_recursive_arguments_preserve_order_shapes_and_dequantized_values() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    for (i, c) in fixtures()["binding_recursive"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = device(c["device"].as_str().unwrap());
        let value = restore(&c["input"], dev);
        let output = unsafe { rt.dequantize_args(&value) }.unwrap();
        drop(value);
        check(&output, &c["output"], dev, &format!("recursive {i}"));
    }
}
#[test]
fn binding_full_tables_mro_first_argument_and_generic_precedence_match_source() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let f = fixtures();
    let t = &f["binding_tables"];
    let mut mros = rt.layout_mros();
    mros.remove("QuantizedLayout");
    assert_eq!(serde_json::json!(mros), t["layouts"]);
    assert_eq!(serde_json::json!(rt.handler_slots()), t["handlers"]);
    assert_eq!(serde_json::json!(rt.generic_slots()), t["generic"]);
    let statuses:serde_json::Map<_,_>=rt.backends.policy.list_backends().into_iter().map(|(k,v)|(k,serde_json::json!({"available":v.available,"disabled":v.disabled,"unavailable_reason":v.unavailable_reason,"capabilities":v.capabilities}))).collect();
    assert_eq!(Value::Object(statuses), t["backends"]);
    for c in f["binding_routing"].as_array().unwrap() {
        for (name, mro) in c["mros"].as_object().unwrap() {
            rt.register_layout(
                name,
                mro.as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap().into())
                    .collect(),
                None,
            );
        }
        for (name, label) in [("Left", "left"), ("Right", "right")] {
            rt.register_handler(
                "aten::mul.Tensor",
                name,
                Arc::new(move |_, _, _| Ok(V::Text(label.into()))),
            );
        }
        rt.register_handler(
            "aten::clone.default",
            "Diamond",
            Arc::new(|_, _, _| panic!("Generic handler must win")),
        );
        let args: Vec<_> = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cpu))
            .collect();
        let kw = kwargs(&c["kwargs"], Device::Cpu);
        let out = unsafe { rt.dispatch(c["operation"].as_str().unwrap(), &args, &kw) }.unwrap();
        check(&out, &c["output"], Device::Cpu, c["mode"].as_str().unwrap());
    }
}
#[test]
fn binding_callable_handles_survive_reregistration_and_gaps_never_change_selection() {
    let rt = Runtime::new(kitchen(), BackendRegistry::new(None));
    let caps = || BTreeMap::from([("f".into(), FunctionConstraints::default())]);
    fn body(s: &'static str) -> BTreeMap<String, Callable> {
        BTreeMap::from([(
            "f".into(),
            Arc::new(move |_: &Runtime, _: &[V], _: &Kwargs| Ok(V::Text(s.into()))) as Callable,
        )])
    }
    rt.backends.register("eager", caps(), body("eager"));
    rt.backends.register("cuda", caps(), body("cuda-old"));
    let mut events = Vec::<Value>::new();
    let run = |f: &imagine_inference::execution::Resolved| {
        let V::Text(s) = unsafe { f.call(&rt, &[], &Vec::new()) }.unwrap() else {
            panic!()
        };
        Value::String(s)
    };
    let held = rt.backends.resolve("f", None, &BTreeMap::new()).unwrap();
    events.push(run(&held));
    rt.backends.register("cuda", caps(), body("cuda-new"));
    events.push(run(&rt
        .backends
        .resolve("f", None, &BTreeMap::new())
        .unwrap()));
    events.push(run(&held));
    rt.backends.policy.disable("cuda");
    events.push(run(&rt
        .backends
        .resolve("f", None, &BTreeMap::new())
        .unwrap()));
    events.push(run(&held));
    let e = rt
        .backends
        .resolve("f", Some("cuda"), &BTreeMap::new())
        .err()
        .unwrap();
    let Error::Selection(e) = e else { panic!() };
    events.push(serde_json::json!({"error":e.exception_class(),"message":e.to_string()}));
    rt.backends.policy.enable("cuda");
    {
        let _guard = rt.backends.policy.use_backend("eager").unwrap();
        events.push(run(&rt
            .backends
            .resolve("f", None, &BTreeMap::new())
            .unwrap()));
    }
    events.push(run(&rt
        .backends
        .resolve("f", None, &BTreeMap::new())
        .unwrap()));
    assert_eq!(
        serde_json::json!(events),
        fixtures()["binding_callables"][0]["events"]
    );
    rt.backends.register("cuda", caps(), BTreeMap::new());
    assert!(
        matches!(rt.backends.resolve("f",None,&BTreeMap::new()),Err(Error::BindingGap{backend,..}) if backend=="cuda")
    );
    assert_eq!(rt.backends.gaps(), vec![("cuda".into(), "f".into())]);
}
