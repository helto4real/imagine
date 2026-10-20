use super::{
    binding_tests::{check, kwargs, reference_error, restore},
    kitchen,
};
use imagine_inference::{execution::Runtime, execution_value::Value as V};
use serde_json::Value;
use std::collections::BTreeMap;
use tch::Device;
#[test]
fn original_geometry_wrappers_preserve_shapes_indices_namedtuple_fields_graph_results_and_errors() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let rt = Runtime::frozen_partial(native).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-mesh.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 169);
    for c in fixture["cases"].as_array().unwrap() {
        let mut args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cuda(0)))
            .collect::<Vec<_>>();
        if let Some(triangles) = c.get("triangles") {
            let bvh = rt
                .backends
                .resolve("mesh_bvh", Some("cuda"), &BTreeMap::new())
                .unwrap();
            let bvh =
                unsafe { bvh.call(&rt, &[restore(triangles, Device::Cuda(0))], &vec![]) }.unwrap();
            args.insert(0, bvh);
        }
        let kw = kwargs(&c["kwargs"], Device::Cuda(0));
        let f = rt
            .backends
            .resolve(c["op"].as_str().unwrap(), Some("cuda"), &BTreeMap::new())
            .unwrap();
        let output = unsafe { f.call(&rt, &args, &kw) };
        if c["error"].is_string() {
            reference_error(
                output
                    .err()
                    .unwrap_or_else(|| panic!("Expected source error {c}")),
                c,
            );
        } else {
            let value = output.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            check(
                &value,
                &c["output"],
                Device::Cuda(0),
                c["label"].as_str().unwrap(),
            );
            if let V::NamedTuple(_) = value {
                assert!(
                    value.same_object(&value.clone()),
                    "NamedTuple clone preserves source object identity"
                );
            }
        }
    }
}

#[test]
fn source_namedtuple_class_identity_condition_equality_and_metadata_diagnostics_match() {
    use imagine_inference::{conditioning, execution_value::NamedTupleClass, registry::*};
    static PAIR_A: NamedTupleClass = NamedTupleClass { name: "Pair" };
    static PAIR_B: NamedTupleClass = NamedTupleClass { name: "Pair" };
    static TRIPLE: NamedTupleClass = NamedTupleClass { name: "Triple" };
    static EMPTY: NamedTupleClass = NamedTupleClass { name: "Empty" };
    fn value(v: &Value) -> V {
        match v["type"].as_str().unwrap() {
            "scalar" => V::Int(v["value"].as_i64().unwrap()),
            "tuple" => V::Tuple(v["values"].as_array().unwrap().iter().map(value).collect()),
            "list" => V::List(v["values"].as_array().unwrap().iter().map(value).collect()),
            "namedtuple" => {
                let class = match v["class_id"].as_str().unwrap() {
                    "pair_a" => &PAIR_A,
                    "pair_b" => &PAIR_B,
                    "triple" => &TRIPLE,
                    "empty" => &EMPTY,
                    _ => unreachable!(),
                };
                let fields = v["fields"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|f| {
                        let key = match f[0].as_str().unwrap() {
                            "a" => "a",
                            "b" => "b",
                            "c" => "c",
                            _ => unreachable!(),
                        };
                        (key, value(&f[1]))
                    })
                    .collect();
                V::named_tuple(class, fields)
            }
            _ => unreachable!(),
        }
    }
    fn validate(actual: Result<(), Failure>, expected: &Value) {
        if expected["error"].is_string() {
            let error = actual.unwrap_err();
            assert_eq!(error.raised.as_deref(), expected["error"].as_str());
            assert_eq!(error.reason, expected["message"].as_str().unwrap());
        } else if expected["success"] == true {
            actual.unwrap();
        } else {
            let error = actual.unwrap_err();
            assert!(error.raised.is_none());
            assert_eq!(error.parameter, expected["parameter"].as_str().unwrap());
            assert_eq!(error.reason, expected["reason"].as_str().unwrap());
        }
    }
    let native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-namedtuple.json")).unwrap();
    let values = fixture["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["key"].as_str().unwrap(), value(&c["value"])))
        .collect::<BTreeMap<_, _>>();
    let constraints = FunctionConstraints {
        params: vec![(
            "x".into(),
            ParamConstraint {
                types: Some(vec![AllowedType::Scalar(ScalarType::Int)]),
                ..Default::default()
            },
        )],
        ..Default::default()
    };
    let na = imagine_inference::backend_contract::constraints("cuda", true)
        .unwrap()
        .remove("na3d")
        .unwrap();
    for c in fixture["values"].as_array().unwrap() {
        let v = &values[c["key"].as_str().unwrap()];
        if matches!(v, V::NamedTuple(_)) {
            assert!(v.same_object(&v.clone()));
        }
        validate(
            constraints.validate(
                &BTreeMap::from([("x".into(), v.metadata(&native).unwrap())]),
                Some((8, 9)),
            ),
            &c["dtype"],
        );
        validate(
            na.validate(
                &BTreeMap::from([("kernel_size".into(), v.metadata(&native).unwrap())]),
                Some((8, 9)),
            ),
            &c["na"],
        );
    }
    assert_eq!(fixture["equality"].as_array().unwrap().len(), 100);
    for c in fixture["equality"].as_array().unwrap() {
        assert_eq!(
            conditioning::is_equal(
                &values[c["a"].as_str().unwrap()],
                &values[c["b"].as_str().unwrap()]
            )
            .unwrap(),
            c["cond_equal"].as_bool().unwrap(),
            "{c}"
        );
    }
}
