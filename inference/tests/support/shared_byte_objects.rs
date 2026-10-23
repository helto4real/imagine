use imagine_inference::{
    checkpoint::builtin_keys::BuiltinKeys,
    execution_value::Value as V,
    native::Error,
    patching::options::{
        DictionaryKeyBindings, GeneralMapping, Options, SourceByteArray as A, SourceBytes as B,
        SourceText,
    },
    python_hash::HashState,
    source_key_error::KeyRepresentation,
};
use serde_json::Value;
use std::{cell::Cell, rc::Rc};
fn bytes(v: &Value) -> Vec<u8> {
    super::exact_hex(v)
}
#[test]
fn native_byte_objects_preserve_copy_and_allocated_caches_hashes_binary_type_mutable_aliases_repr_and_dictionary_keys()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-byte-objects.json")).unwrap();
    let keys = Rc::new(BuiltinKeys {
        hash_state: HashState::from_seed(0),
        decimal_digit_limit: Rc::new(Cell::new(4300)),
    });
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 264);
    for c in cases {
        let data = bytes(&c["data"]);
        let first = B::from_data(&data);
        let second = B::from_data(&data);
        let fresh = B::from_allocated(data.clone());
        let fresh2 = B::from_allocated(data.clone());
        let mutable = A::from_data(data.clone());
        let mutable2 = A::from_data(data.clone());
        assert_eq!(
            first.same_object(&second),
            c["copy_same"].as_bool().unwrap()
        );
        assert_eq!(
            fresh.same_object(&fresh2),
            c["allocated_same"].as_bool().unwrap()
        );
        assert_eq!(
            first.same_object(&fresh),
            c["copy_allocated_same"].as_bool().unwrap()
        );
        assert_eq!(
            mutable.same_object(&mutable2),
            c["mutable_same"].as_bool().unwrap()
        );
        assert_eq!(B::get(&first).unwrap().bytes(), data);
        assert_eq!(&*A::get(&mutable).unwrap().bytes(), data);
        assert_eq!(keys.hash(&first).unwrap(), c["hash"].as_i64().unwrap());
        assert_eq!(!data.is_empty(), c["truth"].as_bool().unwrap());
        assert_eq!(keys.repr(&first).unwrap(), c["repr"].as_str().unwrap());
        assert_eq!(
            keys.repr(&mutable).unwrap(),
            c["mutable_repr"].as_str().unwrap()
        );
        assert_eq!(
            keys.equal(&first, &mutable).unwrap(),
            c["bytes_mutable_equal"].as_bool().unwrap()
        );
        assert_eq!(
            keys.equal(&mutable, &mutable2).unwrap(),
            c["mutable_equal"].as_bool().unwrap()
        );
        let text =
            SourceText::from_codepoints(data.iter().map(|b| u32::from(*b)).collect()).unwrap();
        assert_eq!(
            keys.equal(&first, &text).unwrap(),
            c["text_equal"].as_bool().unwrap()
        );
        match keys.hash(&mutable).unwrap_err() {
            Error::ReferenceException { class, message } => {
                assert_eq!(class, c["mutable_hash"]["error"]["type"].as_str().unwrap());
                assert_eq!(
                    message,
                    c["mutable_hash"]["error"]["message"].as_str().unwrap()
                );
            }
            e => panic!("{e:?}"),
        }
        let d = GeneralMapping::empty(keys.clone());
        for (k, v) in [(&first, 1), (&second, 2), (&fresh, 3)] {
            d.set_item(k.clone(), Options::Atom(V::Int(v))).unwrap();
        }
        let m = GeneralMapping::get(&d).unwrap();
        assert_eq!(
            m.dictionary.len(),
            c["dict_length"].as_u64().unwrap() as usize
        );
        let items = m.dictionary.entries();
        assert_eq!(
            items[0].0.same_object(&first),
            c["dict_key_same"].as_bool().unwrap()
        );
        let Options::Atom(V::Int(v)) = &items[0].1 else {
            panic!("source dict value")
        };
        assert_eq!(Some(*v), c["dict_value"].as_i64());
        assert!(first.snapshot().is_err());
        assert!(mutable.snapshot().is_err());
    }
    let mutable = A::from_data(b"a".to_vec());
    let missing = GeneralMapping::empty(keys)
        .pop_item(&mutable)
        .err()
        .expect("empty pop still raises owned KeyError");
    assert!(missing.is_reference("KeyError"));
    assert_eq!(
        missing.python_message().unwrap(),
        f["mutation"]["before"].as_str().unwrap()
    );
    A::get(&mutable)
        .unwrap()
        .replace_owned_buffer(bytes(&f["mutation"]["data"]));
    assert_eq!(
        missing.python_message().unwrap(),
        f["mutation"]["after"].as_str().unwrap()
    );
    let Error::KeyError(error) = missing.root() else {
        panic!("KeyError")
    };
    let Options::Tuple(args) = error.arguments() else {
        panic!("tuple")
    };
    assert_eq!(
        args[0].same_object(&mutable),
        f["mutation"]["same"].as_bool().unwrap()
    );
}
