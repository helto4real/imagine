use imagine_inference::{
    checkpoint::{builtin_keys::BuiltinKeys, builtin_scalar_ops},
    native::Error,
    patching::options::{
        DictionaryKeyBindings, FrozenSet, GeneralSet, MappingKeyBindings, Options, SourceNumber,
    },
    python_hash::HashState,
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};
fn number(text: &str) -> Options {
    SourceNumber::integer(&text.parse::<i128>().unwrap().to_le_bytes())
}
fn view(set: &FrozenSet) -> Value {
    let order: Vec<_> = set
        .entries()
        .iter()
        .map(|(key, _)| {
            SourceNumber::get(key)
                .unwrap()
                .repr_with_limit(4300)
                .unwrap()
        })
        .collect();
    json!({"order":order,"length":set.len(),"capacity":set.capacity(),"hash":set.hash(),"repeated_hash":set.hash(),"copy_is_self":set.same_object(&set.copy())})
}
#[test]
fn original_frozenset_hash_cache_copy_owners_nested_keys_and_mutable_set_lookup_discard_remove_wrappers()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-frozen-set.json")).unwrap();
    let keys = Rc::new(BuiltinKeys {
        hash_state: HashState::from_seed(0),
        decimal_digit_limit: Rc::new(Cell::new(4300)),
    });
    for (i, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let source = GeneralSet::new(keys.clone());
        for v in c["values"].as_array().unwrap() {
            source.add(&number(v.as_str().unwrap())).unwrap();
        }
        for v in c["removed"].as_array().unwrap() {
            source.discard(&number(v.as_str().unwrap())).unwrap();
        }
        let frozen = FrozenSet::from_set(&source).unwrap();
        assert_eq!(view(&frozen), c["frozen"], "case {i}");
        let copy = FrozenSet::from_set(&source).unwrap();
        assert_eq!(!copy.same_object(&frozen), c["distinct"].as_bool().unwrap());
        let frozen_value = frozen.clone().into_options();
        assert_eq!(
            keys.equal(&frozen_value, &copy.into_options()).unwrap(),
            c["equal"].as_bool().unwrap()
        );
        assert_eq!(
            keys.equal(&frozen_value, &source.clone().into_options())
                .unwrap(),
            c["mutable_equal"].as_bool().unwrap()
        );
        let target = GeneralSet::new(keys.clone());
        target.add(&frozen_value).unwrap();
        let lookup = source.copy_set().unwrap().into_options();
        assert_eq!(
            target.contains(&lookup).unwrap(),
            c["contained"].as_bool().unwrap()
        );
        let wrong = GeneralSet::new(keys.clone());
        wrong.add(&SourceNumber::from_i64(-777)).unwrap();
        let wrong = wrong.into_options();
        let error = target.remove(&wrong).unwrap_err();
        let Error::KeyError(error) = error else {
            panic!("original KeyError")
        };
        let Options::Tuple(args) = error.arguments() else {
            panic!("args")
        };
        assert!(args[0].same_object(&wrong));
        assert_eq!(
            c["missing_error"],
            json!({"type":"KeyError","original":true})
        );
        assert!(target.discard(&lookup).unwrap());
        assert_eq!(target.len(), c["after_discard"].as_u64().unwrap() as usize);
        target.add(&frozen_value).unwrap();
        target.remove(&lookup).unwrap();
        assert_eq!(target.len(), c["after_remove"].as_u64().unwrap() as usize);
        let nested = GeneralSet::new(keys.clone());
        nested.add(&frozen_value).unwrap();
        nested
            .add(
                &FrozenSet::from_set(&GeneralSet::new(keys.clone()))
                    .unwrap()
                    .into_options(),
            )
            .unwrap();
        let one = GeneralSet::new(keys.clone());
        one.add(&SourceNumber::from_i64(913)).unwrap();
        nested
            .add(&FrozenSet::from_set(&one).unwrap().into_options())
            .unwrap();
        let nested = FrozenSet::from_set(&nested).unwrap();
        assert_eq!(nested.hash(), c["nested_hash"].as_i64().unwrap());
        assert_eq!(nested.len(), c["nested_length"].as_u64().unwrap() as usize);
        assert!(target.add(&source.clone().into_options()).is_err());
        // Existing iterators keep their immutable owner and table after caller drops it.
        let iterator = frozen.iterator();
        let expected = frozen.entries();
        drop(frozen);
        let mut actual = vec![];
        while let Some(key) = iterator.next().unwrap() {
            actual.push(key);
        }
        assert_eq!(actual.len(), expected.len());
        for (a, (b, _)) in actual.iter().zip(expected) {
            assert!(keys.equal(a, &b).unwrap());
        }
    }
    let frozen = FrozenSet::from_set(&GeneralSet::new(keys.clone()))
        .unwrap()
        .into_options();
    for (i, args) in [
        vec![frozen.clone()],
        vec![SourceNumber::from_i64(1), frozen],
    ]
    .iter()
    .enumerate()
    {
        let Err(error) = builtin_scalar_ops::complex(args) else {
            panic!("expected complex error")
        };
        let Error::ReferenceException { class, message } = error else {
            panic!("complex error")
        };
        assert_eq!(
            json!({"type":class,"message":message}),
            fixture["complex_errors"][i]
        );
    }
}
#[derive(Clone)]
pub(super) struct CountingKeys {
    pub(super) builtin: BuiltinKeys,
    pub(super) events: Rc<std::cell::RefCell<Vec<Value>>>,
}
impl DictionaryKeyBindings for CountingKeys {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        if SourceNumber::get(key).is_some() {
            self.events.borrow_mut().push(json!([
                "hash",
                SourceNumber::get(key)
                    .unwrap()
                    .repr_with_limit(4300)?
                    .parse::<usize>()
                    .unwrap()
            ]));
            Ok(7)
        } else {
            self.builtin.hash(key)
        }
    }
    fn identity_token(&self, key: &Options) -> u64 {
        self.builtin.identity_token(key)
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        if let (Some(a), Some(b)) = (SourceNumber::get(a), SourceNumber::get(b)) {
            self.events.borrow_mut().push(json!([
                "equal",
                a.repr_with_limit(4300)?.parse::<usize>().unwrap(),
                b.repr_with_limit(4300)?.parse::<usize>().unwrap()
            ]));
        }
        self.builtin.equal(a, b)
    }
    fn exact_unicode(&self, key: &Options) -> bool {
        self.builtin.exact_unicode(key)
    }
}
impl MappingKeyBindings for CountingKeys {
    fn text_key(&self, text: &str) -> Result<Options, Error> {
        self.builtin.text_key(text)
    }
    fn text_contents(&self, key: &Options) -> Result<Option<String>, Error> {
        self.builtin.text_contents(key)
    }
    fn missing_key(&self, key: &Options) -> Result<Error, Error> {
        self.builtin.missing_key(key)
    }
}
#[test]
fn original_frozenset_copy_and_repeated_hash_reuse_entry_hashes_without_key_callbacks() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-frozen-set.json")).unwrap();
    let events = Rc::new(std::cell::RefCell::new(vec![]));
    let keys = Rc::new(CountingKeys {
        builtin: BuiltinKeys {
            hash_state: HashState::from_seed(0),
            decimal_digit_limit: Rc::new(Cell::new(4300)),
        },
        events: events.clone(),
    });
    let source = GeneralSet::new(keys);
    for n in 0..6 {
        source.add(&SourceNumber::from_i64(n)).unwrap();
    }
    events.borrow_mut().clear();
    let frozen = FrozenSet::from_set(&source).unwrap();
    assert_eq!(
        json!(*events.borrow()),
        fixture["cached_hash"]["copy_events"]
    );
    events.borrow_mut().clear();
    let h = frozen.hash();
    assert_eq!(h, frozen.hash());
    assert_eq!(
        json!(*events.borrow()),
        fixture["cached_hash"]["hash_events"]
    );
    assert_eq!(h, fixture["cached_hash"]["value"].as_i64().unwrap());
}
