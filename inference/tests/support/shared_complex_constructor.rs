use imagine_inference::{
    checkpoint::{builtin_keys::BuiltinKeys, builtin_scalar_ops},
    native::Error,
    patching::options::{
        GeneralMapping, Options, SourceByteArray, SourceBytes, SourceNone, SourceNumber as N,
        SourceText,
    },
    python_hash::HashState,
};
use serde_json::Value;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
fn bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn float(s: &str) -> f64 {
    f64::from_le_bytes(bytes(s).try_into().unwrap())
}
fn error(error: Error, expected: &Value) {
    match error {
        Error::ReferenceException { class, message } => {
            assert_eq!(class, expected["type"].as_str().unwrap());
            assert_eq!(message, expected["message"].as_str().unwrap());
        }
        e => panic!("{e:?}"),
    }
}
#[test]
fn builtin_complex_preserves_exact_payloads_aliases_positional_errors_unicode_grammar_and_integer_float_overflow()
 {
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-complex-constructor.json"
    ))
    .unwrap();
    let keys = Rc::new(BuiltinKeys {
        hash_state: HashState::from_seed(0),
        decimal_digit_limit: Rc::new(Cell::new(4300)),
    });
    let values = f["definitions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| match d["kind"].as_str().unwrap() {
            "bytes" => SourceBytes::from_data(&bytes(d["data"].as_str().unwrap())),
            "bytearray" => SourceByteArray::from_data(bytes(d["data"].as_str().unwrap())),
            "none" => SourceNone::value(),
            "bool" => N::boolean(d["value"].as_bool().unwrap()),
            "int" => N::integer(&bytes(d["bytes"].as_str().unwrap())),
            "float" => N::float(float(d["bits"].as_str().unwrap())),
            "complex" => N::complex_parts(
                float(d["real"].as_str().unwrap()),
                float(d["imaginary"].as_str().unwrap()),
            ),
            "str" => SourceText::from_codepoints(
                d["points"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p.as_u64().unwrap() as u32)
                    .collect(),
            )
            .unwrap(),
            "list" => Options::List(Rc::new(RefCell::new(vec![]))),
            "tuple" => Options::Tuple(Rc::new(vec![])),
            "dict" => GeneralMapping::empty(keys.clone()),
            _ => panic!("fixture kind"),
        })
        .collect::<Vec<_>>();
    for (index, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| values[i.as_u64().unwrap() as usize].clone())
            .collect::<Vec<_>>();
        let result = builtin_scalar_ops::complex(&args);
        if c["error"].is_object() {
            error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("expected constructor error {index}")),
                &c["error"],
            );
        } else {
            let out = result.unwrap_or_else(|e| panic!("constructor {index}: {e:?}"));
            let (r, i) = N::get(&out).unwrap().complex_bits().unwrap();
            assert_eq!(
                r,
                float(c["output"]["real"].as_str().unwrap()).to_bits(),
                "complex real case {index}"
            );
            assert_eq!(
                i,
                float(c["output"]["imaginary"].as_str().unwrap()).to_bits(),
                "complex imag case {index}"
            );
            for (i, v) in args.iter().enumerate() {
                assert_eq!(out.same_object(v), c["same"][i].as_bool().unwrap());
            }
        }
    }
    for c in f["conversions"].as_array().unwrap() {
        let n = N::get(&values[c["index"].as_u64().unwrap() as usize]).unwrap();
        let result = n.to_float();
        if c["error"].is_object() {
            error(result.expect_err("float conversion error"), &c["error"]);
        } else {
            assert_eq!(
                result.unwrap().to_bits(),
                float(c["output"].as_str().unwrap()).to_bits()
            );
        }
    }
}
