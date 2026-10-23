use imagine_inference::{
    checkpoint::{builtin_keys::BuiltinKeys, builtin_set_ops},
    native::Error,
    patching::options::{
        FrozenSet, GeneralMapping, GeneralSet, Options, SourceByteArray, SourceBytes, SourceNone,
        SourceNumber, SourceText,
    },
    python_hash::HashState,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
fn keys() -> Rc<BuiltinKeys> {
    Rc::new(BuiltinKeys {
        hash_state: HashState::from_seed(0),
        decimal_digit_limit: Rc::new(Cell::new(4300)),
    })
}
fn no_fallback(_: &GeneralSet, _: &Options) -> Result<(), Error> {
    panic!("exact fixture must not use an iterable fallback")
}
fn materialize(d: &Value, keys: Rc<BuiltinKeys>) -> Options {
    let kind = d[0].as_str().unwrap();
    let raw = &d[1];
    match kind {
        "int" => {
            SourceNumber::integer(&raw.as_str().unwrap().parse::<i128>().unwrap().to_le_bytes())
        }
        "str" => SourceText::decoded_codepoints(
            raw.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
        )
        .unwrap(),
        "None" => SourceNone::value(),
        "bytes" => SourceBytes::from_data(
            &raw.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect::<Vec<_>>(),
        ),
        "bytearray" => SourceByteArray::from_data(
            raw.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect(),
        ),
        "dict" => {
            let map = GeneralMapping::empty(keys.clone());
            for pair in raw.as_array().unwrap() {
                map.set_item(
                    materialize(&pair[0], keys.clone()),
                    materialize(&pair[1], keys.clone()),
                )
                .unwrap();
            }
            map
        }
        _ => {
            let values = raw
                .as_array()
                .unwrap()
                .iter()
                .map(|v| materialize(v, keys.clone()))
                .collect();
            match kind {
                "list" => Options::List(Rc::new(RefCell::new(values))),
                "tuple" => Options::Tuple(Rc::new(values)),
                "set" | "frozenset" => builtin_set_ops::construct(
                    kind == "frozenset",
                    &[Options::List(Rc::new(RefCell::new(values)))],
                    keys,
                    no_fallback,
                )
                .unwrap(),
                _ => panic!("kind"),
            }
        }
    }
}
fn encode(v: &Options) -> Value {
    if let Some(n) = SourceNumber::get(v) {
        json!(["int", n.repr_with_limit(4300).unwrap()])
    } else if let Some(s) = SourceText::get(v) {
        json!(["str", s.codepoints()])
    } else if let Options::Tuple(v) = v {
        json!(["tuple", v.iter().map(encode).collect::<Vec<_>>()])
    } else if let Some(s) = FrozenSet::get(v) {
        json!([
            "frozenset",
            s.entries()
                .iter()
                .map(|(k, _)| encode(k))
                .collect::<Vec<_>>()
        ])
    } else {
        panic!("encode")
    }
}
#[test]
fn original_builtin_set_constructors_preserve_exact_container_paths_dict_cached_hash_presizing_unicode_bytes_errors_and_frozen_idempotence()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-set-constructors.json")).unwrap();
    for (i, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let keys = keys();
        let args = if c["input"].is_null() {
            vec![]
        } else {
            vec![materialize(&c["input"], keys.clone())]
        };
        let class = builtin_set_ops::SetClass::value(c["frozen"].as_bool().unwrap());
        let result = builtin_set_ops::SetClass::get(&class)
            .unwrap()
            .call_positional(&args, keys, no_fallback);
        let (error, result) = match result {
            Err(error) => {
                let Error::ReferenceException { class, message } = error.root() else {
                    panic!("error")
                };
                (json!({"type":class,"message":message}), Value::Null)
            }
            Ok(v) => {
                let (order, length, capacity, hash) = if let Some(set) = GeneralSet::get(&v) {
                    (set.entries(), set.len(), set.capacity(), Value::Null)
                } else {
                    let set = FrozenSet::get(&v).unwrap();
                    (set.entries(), set.len(), set.capacity(), json!(set.hash()))
                };
                (
                    Value::Null,
                    json!({"order":order.iter().map(|(key,_)|encode(key)).collect::<Vec<_>>(),"length":length,"capacity":capacity,"same_owner":args.first().is_some_and(|a|a.same_object(&v)),"hash":hash}),
                )
            }
        };
        assert_eq!(error, c["error"], "error {i}");
        assert_eq!(result, c["result"], "result {i}");
    }
    for (i, frozen) in [false, true].iter().enumerate() {
        let args = [
            Options::Tuple(Rc::new(vec![])),
            Options::Tuple(Rc::new(vec![])),
        ];
        let Err(error) = builtin_set_ops::construct(*frozen, &args, keys(), no_fallback) else {
            panic!("error")
        };
        let Error::ReferenceException { class, message } = error.root() else {
            panic!("error")
        };
        assert_eq!(
            json!({"type":class,"message":message}),
            fixture["argument_errors"][i]
        );
    }
}
#[test]
fn original_exact_dictionary_to_set_reuses_cached_key_hashes_and_presizes_before_collision_comparisons()
 {
    use super::shared_frozen_set_tests::CountingKeys;
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-set-constructors.json")).unwrap();
    let events = Rc::new(RefCell::new(vec![]));
    let keys = Rc::new(CountingKeys {
        builtin: (*keys()).clone(),
        events: events.clone(),
    });
    let source = GeneralMapping::empty(keys.clone());
    for i in 0..6 {
        source
            .set_item(SourceNumber::from_i64(i), SourceNone::value())
            .unwrap();
    }
    events.borrow_mut().clear();
    let result = builtin_set_ops::construct(false, &[source], keys, no_fallback).unwrap();
    let result = GeneralSet::get(&result).unwrap();
    assert_eq!(json!(*events.borrow()), fixture["cached_dict"]["events"]);
    assert_eq!(
        json!(
            result
                .entries()
                .iter()
                .map(|(k, _)| SourceNumber::get(k)
                    .unwrap()
                    .repr_with_limit(4300)
                    .unwrap()
                    .parse::<usize>()
                    .unwrap())
                .collect::<Vec<_>>()
        ),
        fixture["cached_dict"]["order"]
    );
    assert_eq!(json!(result.capacity()), fixture["cached_dict"]["capacity"]);
}
#[test]
fn native_builtin_class_owners_preserve_cached_identity_pointer_hash_equality_representation_and_exact_type_errors()
 {
    use imagine_inference::{
        checkpoint::builtin_scalar_ops::ComplexClass, patching::options::DictionaryKeyBindings,
        python_hash, source_key_error::KeyRepresentation,
    };
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-set-constructors.json")).unwrap();
    let keys = keys();
    let classes = [
        builtin_set_ops::SetClass::value(false),
        builtin_set_ops::SetClass::value(true),
        ComplexClass::value(),
    ];
    let copies = [
        builtin_set_ops::SetClass::value(false),
        builtin_set_ops::SetClass::value(true),
        ComplexClass::value(),
    ];
    for (i, class) in classes.iter().enumerate() {
        assert!(class.same_object(&copies[i]));
        assert_eq!(
            keys.hash(class).unwrap(),
            python_hash::identity_hash_address(keys.identity_token(class) as usize)
        );
        assert_eq!(
            json!(keys.repr(class).unwrap()),
            fixture["classes"][i]["repr"]
        );
        let Err(error) =
            imagine_inference::checkpoint::builtin_scalar_ops::complex(std::slice::from_ref(class))
        else {
            panic!("complex error")
        };
        let Error::ReferenceException {
            class: kind,
            message,
        } = error.root()
        else {
            panic!("error")
        };
        assert_eq!(
            json!({"type":kind,"message":message}),
            fixture["classes"][i]["complex_error"]
        );
        let Err(error) = builtin_set_ops::construct(
            false,
            std::slice::from_ref(class),
            keys.clone(),
            no_fallback,
        ) else {
            panic!("iterable error")
        };
        let Error::ReferenceException {
            class: kind,
            message,
        } = error.root()
        else {
            panic!("error")
        };
        assert_eq!(
            json!({"type":kind,"message":message}),
            fixture["classes"][i]["iterable_error"]
        );
        for (j, other) in copies.iter().enumerate() {
            assert_eq!(keys.equal(class, other).unwrap(), i == j);
        }
    }
    let globals = GeneralSet::new(keys.clone());
    for class in classes {
        globals.add(&class).unwrap();
    }
    assert_eq!(globals.len(), 3);
    for class in copies {
        assert!(globals.contains(&class).unwrap());
    }
}
