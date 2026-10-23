use imagine_inference::{
    checkpoint::builtin_keys::BuiltinKeys,
    native::Error,
    patching::options::{GeneralMapping, Options, SourceNone, SourceNumber as N, SourceText},
    python_hash::HashState,
    source_exception::Frame,
    source_key_error::{KeyError, KeyRepresentation},
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn float(s: &str) -> f64 {
    f64::from_le_bytes(hex(s).try_into().unwrap())
}
fn key(v: &Value) -> Options {
    match v["kind"].as_str().unwrap() {
        "none" => SourceNone::value(),
        "bool" => N::boolean(v["value"].as_bool().unwrap()),
        "int" => N::integer(&hex(v["bytes"].as_str().unwrap())),
        "float" => N::float(float(v["bits"].as_str().unwrap())),
        "complex" => N::complex_parts(
            float(v["real"].as_str().unwrap()),
            float(v["imaginary"].as_str().unwrap()),
        ),
        "str" => SourceText::from_codepoints(
            v["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
        )
        .unwrap(),
        _ => panic!("fixture key"),
    }
}
fn check_result(result: Result<String, Error>, expected: &Value) {
    if expected["error"].is_object() {
        match result.expect_err("source formatting exception") {
            Error::ReferenceException { class, message } => {
                assert_eq!(class, expected["error"]["type"].as_str().unwrap());
                assert_eq!(message, expected["error"]["message"].as_str().unwrap());
            }
            e => panic!("{e:?}"),
        }
    } else {
        assert_eq!(result.unwrap(), expected["value"].as_str().unwrap());
    }
}
#[test]
fn native_scalar_keyerrors_preserve_args_identity_exact_repr_and_deferred_mutable_decimal_limit() {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-key-error.json")).unwrap();
    let limit = Rc::new(Cell::new(4300));
    let bindings = Rc::new(BuiltinKeys {
        hash_state: HashState::from_seed(0),
        decimal_digit_limit: limit.clone(),
    });
    assert_eq!(f["default_digit_limit"], 4300);
    assert_eq!(f["cases"].as_array().unwrap().len(), 4230);
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        let k = key(&c["key"]);
        let map = GeneralMapping::empty(bindings.clone());
        let error = map.item(&k).err().expect("missing key");
        assert!(error.is_reference("KeyError"));
        let Error::KeyError(e) = error.root() else {
            panic!("owned KeyError")
        };
        let args = e.arguments();
        assert_eq!(
            args.same_object(&e.arguments()),
            c["args_same"].as_bool().unwrap()
        );
        let Options::Tuple(args) = args else {
            panic!("args tuple")
        };
        assert_eq!(args[0].same_object(&k), c["key_same"].as_bool().unwrap());
        assert_eq!(
            bindings.repr(&k).unwrap(),
            c["representation"].as_str().unwrap(),
            "key repr {i}"
        );
        assert_eq!(
            error.python_message().unwrap(),
            c["message"].as_str().unwrap(),
            "KeyError str {i}"
        );
    }
    for c in f["limit_cases"].as_array().unwrap() {
        let k = key(&c["key"]);
        limit.set(640);
        let error = GeneralMapping::empty(bindings.clone())
            .item(&k)
            .err()
            .unwrap();
        assert!(error.is_reference("KeyError"));
        let args = match error.root() {
            Error::KeyError(e) => e.arguments(),
            _ => panic!("keyerror"),
        };
        for h in c["history"].as_array().unwrap() {
            limit.set(h["limit"].as_u64().unwrap() as usize);
            check_result(error.python_message(), &h["format"]);
            assert!(error.is_reference(h["class_unchanged"].as_str().unwrap()));
            let Options::Tuple(args) = &args else {
                panic!("tuple")
            };
            assert_eq!(args[0].same_object(&k), h["args_same"].as_bool().unwrap());
        }
    }
}
struct Own {
    fail: Cell<bool>,
}
struct Representation {
    events: Rc<RefCell<Vec<&'static str>>>,
}
impl KeyRepresentation for Representation {
    fn repr(&self, key: &Options) -> Result<String, Error> {
        self.events.borrow_mut().push("repr");
        let Options::Object(key) = key else {
            panic!("own repr receiver")
        };
        let key = key.downcast_ref::<Own>().unwrap();
        if key.fail.get() {
            Err(Error::ReferenceException {
                class: "ValueError",
                message: "own repr failed".into(),
            })
        } else {
            Ok("<own key>".into())
        }
    }
}
#[test]
fn native_keyerror_owns_deferred_repr_argument_and_releases_exposed_args_independently_of_traceback_root()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-key-error.json")).unwrap();
    let c = &f["own_protocol"];
    let own = Rc::new(Own {
        fail: Cell::new(false),
    });
    let weak = Rc::downgrade(&own);
    let events = Rc::new(RefCell::new(vec![]));
    let key = Options::Object(own);
    let e = KeyError::new(
        key.clone(),
        Rc::new(Representation {
            events: events.clone(),
        }),
    );
    let args = e.arguments();
    let Options::Tuple(v) = &args else {
        panic!("tuple")
    };
    assert!(v[0].same_object(&key));
    drop(key);
    assert_eq!(json!(*events.borrow()), c["before"]);
    let frame = Frame::new("own scalar error caller");
    frame.finish();
    let error = Error::KeyError(e.clone()).retain_frame(frame);
    assert!(error.is_reference("KeyError"));
    assert_eq!(
        error.python_message().unwrap(),
        c["first"].as_str().unwrap()
    );
    weak.upgrade().unwrap().fail.set(true);
    check_result(error.python_message(), &c["failed"]);
    assert_eq!(
        format!("KeyError: {error}\n"),
        c["diagnostic"].as_str().unwrap()
    );
    assert_eq!(json!(*events.borrow()), c["events"]);
    e.replace_arguments(Options::Tuple(Rc::new(vec![])))
        .unwrap();
    assert_eq!(
        weak.upgrade().is_some(),
        c["alive_with_exposed_args"].as_bool().unwrap()
    );
    drop(args);
    assert_eq!(
        weak.upgrade().is_none(),
        c["expired_after_release"].as_bool().unwrap()
    );
    assert_eq!(
        error.python_message().unwrap(),
        c["empty_message"].as_str().unwrap()
    );
    assert!(matches!(error.root(),Error::KeyError(root)if Rc::ptr_eq(root,&e)));
}
