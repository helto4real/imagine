use imagine_inference::{
    checkpoint::{
        builtin_keys::BuiltinKeys,
        builtin_scalar_ops,
        dtype_objects::SourceDtype,
        weights_only_globals::{self, Bindings},
    },
    execution_value::Value as V,
    memory::dtype::{RegisteredDtypeSizeBindings, dtype_size},
    native::Error,
    patching::options::{DictionaryKeyBindings, GeneralMapping, Options, SourceNumber, SourceText},
    python_hash::{HashState, identity_hash_address},
    source_key_error::KeyRepresentation,
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};
fn result<T>(value: Result<T, Error>, encode: impl FnOnce(T) -> Value) -> Value {
    match value {
        Ok(value) => json!({"value":encode(value)}),
        Err(error) => {
            let Error::ReferenceException { class, .. } = error.root() else {
                panic!("{error:?}")
            };
            json!({"error":{"type":class,"message":error.python_message().unwrap()}})
        }
    }
}
fn primitive(value: Options) -> Value {
    match value.snapshot().unwrap() {
        V::Int(v) => json!(v),
        V::Bool(v) => json!(v),
        V::Text(v) => json!(v),
        _ => panic!("typed scalar property"),
    }
}
struct Registry {
    values: Options,
    calls: usize,
}
impl Bindings for Registry {
    fn allowed(&mut self) -> Result<Options, Error> {
        self.calls += 1;
        Ok(self.values.clone())
    }
    fn user(&mut self) -> Result<Options, Error> {
        Ok(Options::dictionary(vec![]))
    }
}
#[test]
fn concrete_torch_dtype_owners_cover_all_scalar_codes_aliases_properties_original_errors_conversions_keys_and_global_membership_without_kind_coercion()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dtype-objects.json")).unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 46);
    let keys = Rc::new(BuiltinKeys {
        hash_state: HashState::from_seed(0),
        decimal_digit_limit: Rc::new(Cell::new(4300)),
    });
    let map = GeneralMapping::empty(keys.clone());
    let mut registry = Registry {
        values: GeneralMapping::empty(keys.clone()),
        calls: 0,
    };
    // This is the dtype membership component, not the complete default table
    // or a replacement for its source set/registry construction order.
    for code in 0..46 {
        let value = SourceDtype::from_scalar_type(code).unwrap();
        let dtype = SourceDtype::get(&value).unwrap();
        registry
            .values
            .set(&dtype.representation(), value.clone())
            .unwrap();
        map.set_item(value, SourceNumber::from_i64(i64::from(code)))
            .unwrap();
    }
    for c in cases {
        let code = i32::try_from(c["code"].as_i64().unwrap()).unwrap();
        let value = SourceDtype::from_scalar_type(code).unwrap();
        let dtype = SourceDtype::get(&value).unwrap();
        assert_eq!(dtype.scalar_type(), code);
        assert_eq!(dtype.name(), c["name"].as_str().unwrap());
        assert_eq!(dtype.representation(), c["repr"]);
        assert_eq!(keys.repr(&value).unwrap(), c["repr"]);
        assert_eq!(primitive(dtype.reduce_name()), c["reduce"]);
        assert!(value.same_object(&SourceDtype::lookup(c["name"].as_str().unwrap()).unwrap()));
        assert_eq!(
            dtype.hash(),
            identity_hash_address(dtype.identity_token() as usize)
        );
        assert_eq!(keys.hash(&value).unwrap(), dtype.hash());
        for (name, expected) in c["properties"].as_object().unwrap() {
            assert_eq!(
                result(dtype.property(name), primitive),
                *expected,
                "dtype {code} property {name}"
            );
        }
        let real = dtype.to_real();
        assert_eq!(
            SourceDtype::get(&real).unwrap().representation(),
            c["to_real"]["value"]
        );
        assert_eq!(
            result(dtype.to_complex(), |value| json!(
                SourceDtype::get(&value).unwrap().representation()
            )),
            c["to_complex"],
            "dtype {code}"
        );
        assert_eq!(
            result(
                builtin_scalar_ops::complex(std::slice::from_ref(&value)),
                |_| panic!("dtype cannot convert to complex")
            ),
            c["complex_first"]
        );
        assert_eq!(
            result(
                builtin_scalar_ops::complex(&[SourceNumber::from_i64(1), value.clone()]),
                |_| panic!("dtype cannot convert to complex")
            ),
            c["complex_second"]
        );
        assert_eq!(
            primitive(dtype_size(&value, &RegisteredDtypeSizeBindings).unwrap()),
            c["properties"]["itemsize"]["value"]
        );
        if let Some(kind) = dtype.kind() {
            assert!(matches!(value.snapshot().unwrap(),V::DType(v)if v==kind));
        } else {
            assert!(matches!(value.snapshot(), Err(Error::BindingGap { .. })));
        }
        for alias in c["aliases"].as_array().unwrap() {
            assert!(value.same_object(&SourceDtype::lookup(alias.as_str().unwrap()).unwrap()));
        }
        for (alias, expected) in c["alias_loads"].as_object().unwrap() {
            assert_eq!(
                result(
                    weights_only_globals::resolve("torch", alias, &mut registry),
                    |value| json!(SourceDtype::get(&value).unwrap().representation())
                ),
                *expected,
                "dtype {code} global {alias}"
            );
        }
        for path in c["global_paths"].as_array().unwrap() {
            let (module, name) = path.as_str().unwrap().rsplit_once('.').unwrap();
            let calls = registry.calls;
            let loaded = weights_only_globals::resolve(module, name, &mut registry).unwrap();
            assert!(loaded.same_object(&value));
            assert_eq!(registry.calls - calls, 2);
        }
        assert_eq!(primitive(map.item(&value).unwrap()), json!(code));
        assert!(
            !keys
                .equal(&value, &SourceNumber::from_i64(i64::from(code)))
                .unwrap()
        );
        assert!(
            !keys
                .equal(&value, &SourceText::from_utf8(dtype.name()))
                .unwrap()
        );
        for other in cases {
            let other_code = i32::try_from(other["code"].as_i64().unwrap()).unwrap();
            let other = SourceDtype::from_scalar_type(other_code).unwrap();
            assert_eq!(keys.equal(&value, &other).unwrap(), code == other_code);
            assert_eq!(value.same_object(&other), code == other_code);
        }
    }
    assert!(SourceDtype::from_scalar_type(-1).is_none());
    assert!(SourceDtype::from_scalar_type(46).is_none());
    assert!(SourceDtype::lookup("FLOAT32").is_none());
}
