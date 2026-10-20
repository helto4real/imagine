use super::binding_tests::reference_error;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        extensions as e,
        options::{self as o, Options},
    },
};
use serde_json::{Value, json};
use std::{rc::Rc, sync::Mutex};

fn value(json: &Value) -> V {
    match json {
        Value::Null => V::None,
        Value::Bool(b) => V::Bool(*b),
        Value::Number(n) => V::Int(n.as_i64().unwrap()),
        Value::String(s) => V::Text(s.clone()),
        Value::Array(a) => V::List(a.iter().map(value).collect()),
        Value::Object(o) => V::Dict(o.iter().map(|(k, v)| (k.clone(), value(v))).collect()),
    }
}
fn json_value(value: &V) -> Value {
    match value {
        V::None => Value::Null,
        V::Bool(b) => json!(b),
        V::Int(n) => json!(n),
        V::Text(s) => json!(s),
        V::List(a) | V::Tuple(a) => Value::Array(a.iter().map(json_value).collect()),
        V::Dict(d) => Value::Object(d.iter().map(|(k, v)| (k.clone(), json_value(v))).collect()),
        _ => panic!("Unexpected synthetic value"),
    }
}
fn number(v: &V) -> i64 {
    if let V::Int(n) = v {
        *n
    } else {
        panic!("Expected integer")
    }
}
fn offset(kwargs: &[(String, V)]) -> i64 {
    kwargs
        .iter()
        .find(|(k, _)| k == "offset")
        .map_or(0, |(_, v)| number(v))
}
#[test]
fn shared_wrapper_registry_stack_and_nested_option_aliases_match_original() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-extensions.json")).unwrap();
    assert_eq!(json!(e::CALLBACKS), fixture["callbacks"]);
    assert_eq!(json!(e::WRAPPERS), fixture["wrappers_registry"]);
    for case in fixture["registries"].as_array().unwrap() {
        let mut registry = e::OrderedRegistry::default();
        for (key, n) in [
            (None, 1),
            (Some("b"), 2),
            (Some("a"), 3),
            (Some("b"), 4),
            (None, 5),
        ] {
            registry.add("own-type", key, Rc::new(n));
        }
        let numbers = |a: Vec<Rc<i64>>| a.iter().map(|v| **v).collect::<Vec<_>>();
        assert_eq!(json!(numbers(registry.all("own-type"))), case["all"]);
        assert_eq!(
            json!(numbers(registry.with_key("own-type", None))),
            case["none"]
        );
        assert_eq!(
            json!(numbers(registry.with_key("own-type", Some("b")))),
            case["b"]
        );
        assert_eq!(
            json!(numbers(registry.with_key("missing", None))),
            case["missing"]
        );
        let mut copy = registry.clone();
        copy.add("own-type", None, Rc::new(6));
        assert_eq!(numbers(registry.all("own-type")), [1, 5, 2, 4, 3]);
    }
    for case in fixture["wrappers"].as_array().unwrap() {
        let trace = Rc::new(Mutex::new(Vec::<Value>::new()));
        let original: e::Original = {
            let trace = trace.clone();
            Rc::new(move |args, kwargs| {
                let x = number(&args[0]);
                let offset = offset(kwargs);
                trace.lock().unwrap().push(json!(["original", x, offset]));
                Ok(V::Int(x * 3 + offset))
            })
        };
        let first: e::Wrapper = {
            let trace = trace.clone();
            Rc::new(move |ex, args, kwargs| {
                let x = number(&args[0]);
                let class = ex
                    .class_obj
                    .as_ref()
                    .map(|o| o.downcast_ref::<String>().unwrap().as_str());
                trace
                    .lock()
                    .unwrap()
                    .push(json!(["before-first", ex.idx, class, x]));
                let out = number(&ex.next(
                    &[V::Int(x + 1)],
                    &vec![("offset".into(), V::Int(offset(kwargs) + 2))],
                )?);
                trace.lock().unwrap().push(json!(["after-first", out]));
                Ok(V::Int(out + 10))
            })
        };
        let second: e::Wrapper = {
            let trace = trace.clone();
            Rc::new(move |ex, args, kwargs| {
                let x = number(&args[0]);
                let class = ex
                    .class_obj
                    .as_ref()
                    .map(|o| o.downcast_ref::<String>().unwrap().as_str());
                trace
                    .lock()
                    .unwrap()
                    .push(json!(["before-second", ex.idx, class, x]));
                let out = number(&ex.next(&[V::Int(x + 2)], kwargs)?)
                    + number(&ex.next(&[V::Int(x + 3)], kwargs)?);
                trace.lock().unwrap().push(json!(["after-second", out]));
                Ok(V::Int(out + 20))
            })
        };
        let class = case["class_obj"]
            .as_str()
            .map(|s| Rc::new(s.to_owned()) as Rc<dyn std::any::Any>);
        let ex = e::WrapperExecutor::new(
            original,
            class,
            &[first, second],
            case["idx"].as_u64().unwrap() as usize,
        );
        let out = ex.execute(&[V::Int(4)], &vec![("offset".into(), V::Int(5))]);
        if case["error"].is_string() {
            reference_error(out.err().unwrap(), case);
        } else {
            assert_eq!(json!(number(&out.unwrap())), case["output"]);
        }
        assert_eq!(json!(*trace.lock().unwrap()), case["trace"]);
    }
    let last = e::WrapperExecutor::new(Rc::new(|_, _| Ok(V::None)), None, &[], 0);
    reference_error(
        last.next(&[], &vec![]).err().unwrap(),
        &fixture["next_error"],
    );
    for case in fixture["options"].as_array().unwrap() {
        let left = Options::from_value(value(&case["left_before"]));
        let right = Options::from_value(value(&case["right"]));
        let list = left.get("list").unwrap().unwrap();
        let Options::List(ref items) = list else {
            unreachable!()
        };
        let child = items.borrow()[0].clone();
        let out =
            o::merge_nested_dicts(&left, &right, case["copy_left"].as_bool().unwrap()).unwrap();
        assert_eq!(json_value(&out.snapshot().unwrap()), case["output"]);
        assert_eq!(json_value(&left.snapshot().unwrap()), case["left_after"]);
        assert_eq!(json!(out.same_object(&left)), case["root_same"]);
        assert_eq!(
            json!(
                out.get("nested")
                    .unwrap()
                    .unwrap()
                    .same_object(&left.get("nested").unwrap().unwrap())
            ),
            case["nested_same"]
        );
        let output_list = out.get("list").unwrap().unwrap();
        assert_eq!(
            json!(output_list.same_object(&left.get("list").unwrap().unwrap())),
            case["list_same"]
        );
        let Options::List(items) = output_list else {
            unreachable!()
        };
        assert_eq!(
            json!(items.borrow()[0].same_object(&child)),
            case["child_same"]
        );
    }
    for case in fixture["collisions"].as_array().unwrap() {
        let left = Options::from_value(value(&case["left"]));
        let right = Options::from_value(value(&case["right"]));
        reference_error(
            o::merge_nested_dicts(&left, &right, true).err().unwrap(),
            case,
        );
    }
    // Lists are copied one level; nested list children and tuples retain identity.
    let input = Options::from_value(V::Dict(vec![
        (
            "nested".into(),
            V::Dict(vec![(
                "values".into(),
                V::List(vec![V::List(vec![V::Int(1), V::Int(2)])]),
            )]),
        ),
        ("tuple".into(), V::Tuple(vec![V::Int(1), V::Int(2)])),
    ]));
    let copy = o::copy_nested_dicts(&input).unwrap();
    let original_nested = input.get("nested").unwrap().unwrap();
    let copied_nested = copy.get("nested").unwrap().unwrap();
    let original_list = original_nested.get("values").unwrap().unwrap();
    let copied_list = copied_nested.get("values").unwrap().unwrap();
    let (Options::List(a), Options::List(b)) = (&original_list, &copied_list) else {
        unreachable!()
    };
    let aliases = json!({"root_same":copy.same_object(&input),"nested_same":copied_nested.same_object(&original_nested),"list_same":copied_list.same_object(&original_list),"child_same":a.borrow()[0].same_object(&b.borrow()[0]),"tuple_same":copy.get("tuple").unwrap().unwrap().same_object(&input.get("tuple").unwrap().unwrap())});
    assert_eq!(aliases, fixture["copy_aliases"]);
    let _: Option<Error> = None;
}
