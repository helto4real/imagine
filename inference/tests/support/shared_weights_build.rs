use imagine_inference::{
    checkpoint::{
        scalar_bindings::ScalarBindings,
        weights_only::{Bindings as VmBindings, ByteStream, Unpickler},
        weights_only_build::{self as policy, Bindings},
        weights_only_globals, weights_only_reduce,
    },
    execution_value::Value as V,
    native::Error,
    patching::options::{ItemsIterator, Options, SourceNone, SourceNumber},
    python_hash::HashState,
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};
struct Token(&'static str);
struct Slots {
    iterator: ItemsIterator,
    index: Cell<usize>,
}
struct World<'a> {
    mode: &'a str,
    events: Vec<Value>,
    class: Options,
    instance: Options,
    other: Options,
    attrs: Options,
    reads: usize,
}
fn token(label: &'static str) -> Options {
    Options::Object(Rc::new(Token(label)))
}
fn error(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn gap() -> Error {
    Error::BindingGap {
        backend: "build_fixture".into(),
        operation: "outside the admitted own state objects".into(),
    }
}
fn decode(v: &Value) -> Options {
    match v["kind"].as_str().unwrap() {
        "none" => SourceNone::value(),
        "int" => SourceNumber::from_i64(v["value"].as_i64().unwrap()),
        "str" => Options::Atom(V::Text(v["value"].as_str().unwrap().into())),
        "dict" => Options::dictionary(
            v["entries"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| (pair[0]["value"].as_str().unwrap().into(), decode(&pair[1])))
                .collect(),
        ),
        "tuple" => Options::Tuple(Rc::new(
            v["items"].as_array().unwrap().iter().map(decode).collect(),
        )),
        "list" => Options::List(Rc::new(std::cell::RefCell::new(
            v["items"].as_array().unwrap().iter().map(decode).collect(),
        ))),
        _ => panic!("own fixture type"),
    }
}
fn encode(v: V) -> Value {
    match v {
        V::None => json!({"kind":"none"}),
        V::Int(v) => json!({"kind":"int","value":v}),
        V::Text(v) => json!({"kind":"str","value":v}),
        V::Dict(v) => {
            json!({"kind":"dict","entries":v.into_iter().map(|(k,v)|json!([encode(V::Text(k)),encode(v)])).collect::<Vec<_>>()})
        }
        V::Tuple(v) => {
            json!({"kind":"tuple","items":v.into_iter().map(encode).collect::<Vec<_>>()})
        }
        V::List(v) => json!({"kind":"list","items":v.into_iter().map(encode).collect::<Vec<_>>()}),
        _ => panic!("own value"),
    }
}
impl weights_only_globals::Bindings for World<'_> {
    fn allowed(&mut self) -> Result<Options, Error> {
        self.events.push(json!(["allowed"]));
        if self.mode == "allowed-error" {
            return Err(error("ValueError", "own allowed getter failure"));
        }
        Ok(Options::dictionary(if self.mode == "missing" {
            vec![]
        } else {
            vec![("own".into(), self.class.clone())]
        }))
    }
    fn user(&mut self) -> Result<Options, Error> {
        self.events.push(json!(["user"]));
        if self.mode == "user-error" {
            return Err(error("ValueError", "own user getter failure"));
        }
        Ok(Options::dictionary(if self.mode == "user" {
            vec![("own".into(), self.class.clone())]
        } else {
            vec![]
        }))
    }
}
impl weights_only_reduce::Bindings for World<'_> {
    fn equal(&mut self, _: &Options, _: &Options) -> Result<bool, Error> {
        Err(gap())
    }
    fn format(&mut self, value: &Options) -> Result<String, Error> {
        assert!(value.same_object(&self.class));
        self.events.push(json!(["format-class"]));
        Ok("<own class>".into())
    }
    fn bound_self(&mut self, _: &Options) -> Result<Options, Error> {
        Err(gap())
    }
    fn is_attribute_error(&self, error: &Error) -> bool {
        error.is_reference("AttributeError") || error.is_reference("OwnAttributeError")
    }
    fn call(&mut self, _: &Options, _: &Options) -> Result<Options, Error> {
        Err(gap())
    }
    fn tensor_class_contains(&mut self, _: &Options) -> Result<bool, Error> {
        Err(gap())
    }
    fn module_contains_sparse(&mut self, _: &Options) -> Result<bool, Error> {
        Err(gap())
    }
    fn append_sparse(&mut self, _: &Options) -> Result<(), Error> {
        Err(gap())
    }
}
impl Bindings for World<'_> {
    fn exact_type(&mut self, instance: &Options) -> Result<Options, Error> {
        assert!(instance.same_object(&self.instance));
        Ok(self.class.clone())
    }
    fn tensor_type(&mut self) -> Result<Options, Error> {
        Ok(if self.mode.starts_with("tensor") {
            &self.class
        } else {
            &self.other
        }
        .clone())
    }
    fn parameter_type(&mut self) -> Result<Options, Error> {
        Ok(if self.mode.starts_with("parameter") {
            &self.class
        } else {
            &self.other
        }
        .clone())
    }
    fn ordered_dictionary_type(&mut self) -> Result<Options, Error> {
        Ok(if self.mode.starts_with("ordered") {
            &self.class
        } else {
            &self.other
        }
        .clone())
    }
    fn legacy_tensor_set(&mut self, instance: &Options, state: &Options) -> Result<(), Error> {
        assert!(instance.same_object(&self.instance));
        self.events.push(json!(["legacy-get"]));
        if self.mode == "tensor-get-error" {
            return Err(error("ValueError", "own legacy getter failure"));
        }
        let Options::Tuple(values) = state else {
            return Err(gap());
        };
        self.events.push(json!(["legacy-call", values.len()]));
        if self.mode == "tensor-call-error" {
            return Err(error("RuntimeError", "own legacy call failure"));
        }
        self.attrs
            .set("legacy", Options::Atom(V::Int(values.len() as i64)))
    }
    fn get_setstate(&mut self, instance: &Options) -> Result<Options, Error> {
        assert!(instance.same_object(&self.instance));
        self.reads += 1;
        self.events.push(json!(["setstate-get", self.reads]));
        if matches!(self.mode, "setstate-get-error" | "parameter-get-error") {
            return Err(error("RuntimeError", "own setstate getter failure"));
        }
        if self.mode == "attribute-subclass" {
            return Err(error("OwnAttributeError", "own subclass absent"));
        }
        if !matches!(
            self.mode,
            "setstate"
                | "setstate-changes"
                | "setstate-disappears"
                | "setstate-call-error"
                | "parameter"
                | "parameter-call-error"
        ) || self.mode == "setstate-disappears" && self.reads == 2
        {
            return Err(error("AttributeError", "own setstate absent"));
        }
        Ok(token(
            if self.mode == "setstate-changes" && self.reads == 2 {
                "second"
            } else {
                "first"
            },
        ))
    }
    fn call_setstate(&mut self, method: &Options, state: &Options) -> Result<(), Error> {
        let Options::Object(method) = method else {
            return Err(gap());
        };
        let label = method.downcast_ref::<Token>().unwrap().0;
        self.events.push(json!(["setstate-call", label]));
        if matches!(self.mode, "setstate-call-error" | "parameter-call-error") {
            return Err(error("ValueError", "own setstate call failure"));
        }
        self.attrs.set("seen", state.clone())
    }
    fn dictionary_update(&mut self, instance: &Options, state: &Options) -> Result<(), Error> {
        assert!(instance.same_object(&self.instance));
        self.events.push(json!(["dict-get"]));
        if self.mode == "dict-get-error" {
            return Err(error("ValueError", "own dictionary getter failure"));
        }
        self.events.push(json!(["dict-update"]));
        if matches!(self.mode, "dict-update-error" | "ordered-update-error") {
            return Err(error("RuntimeError", "own dictionary update failure"));
        }
        let items = state.items_iterator()?;
        while let Some(pair) = items.next_item()? {
            let Options::Tuple(pair) = pair else {
                unreachable!()
            };
            self.attrs.set_item(pair[0].clone(), pair[1].clone())?;
        }
        Ok(())
    }
    fn split_slot_state(&mut self, state: &Options) -> Result<Option<(Options, Options)>, Error> {
        Ok(match state {
            Options::Tuple(v) if v.len() == 2 => Some((v[0].clone(), v[1].clone())),
            _ => None,
        })
    }
    fn truth(&mut self, value: &Options) -> Result<bool, Error> {
        if SourceNone::get(value).is_some() {
            return Ok(false);
        }
        Ok(match value.snapshot()? {
            V::None => false,
            V::Int(v) => v != 0,
            V::Dict(v) => !v.is_empty(),
            V::Tuple(v) | V::List(v) => !v.is_empty(),
            _ => return Err(gap()),
        })
    }
    fn slot_items(&mut self, slots: &Options) -> Result<Options, Error> {
        self.events.push(json!(["slot-items"]));
        if self.mode == "slot-items-error" {
            return Err(error("RuntimeError", "own slot items failure"));
        }
        Ok(Options::Object(Rc::new(Slots {
            iterator: slots.items_iterator()?,
            index: Cell::new(0),
        })))
    }
    fn next_slot(&mut self, iterator: &Options) -> Result<Option<(Options, Options)>, Error> {
        let Options::Object(iterator) = iterator else {
            return Err(gap());
        };
        let slots = iterator.downcast_ref::<Slots>().unwrap();
        let Some(pair) = slots.iterator.next_item()? else {
            self.events.push(json!(["slot-end"]));
            return Ok(None);
        };
        let index = slots.index.get();
        slots.index.set(index + 1);
        self.events.push(json!(["slot-next", index]));
        if self.mode == "slot-next-error" && index == 1 {
            return Err(error("RuntimeError", "own slot iterator failure"));
        }
        let Options::Tuple(pair) = pair else {
            unreachable!()
        };
        Ok(Some((pair[0].clone(), pair[1].clone())))
    }
    fn set_attribute(
        &mut self,
        instance: &Options,
        key: &Options,
        value: &Options,
    ) -> Result<(), Error> {
        assert!(instance.same_object(&self.instance));
        let Options::Atom(V::Text(key)) = key else {
            return Err(gap());
        };
        self.events.push(json!(["setattr", key]));
        if self.mode == "setattr-error" && key == "fail" {
            return Err(error("RuntimeError", "own setter failure"));
        }
        self.attrs.set(key, value.clone())
    }
}
impl VmBindings for World<'_> {
    fn global(&mut self, _: &[u8], _: &[u8]) -> Result<Options, Error> {
        Err(gap())
    }
    fn new_object(&mut self, _: &Options, _: &Options) -> Result<Options, Error> {
        Err(gap())
    }
    fn reduce(&mut self, _: &Options, _: &Options) -> Result<Options, Error> {
        Err(gap())
    }
    fn build(&mut self, instance: &Options, state: &Options) -> Result<(), Error> {
        policy::build(instance, state, self)
    }
    fn set_item(&mut self, _: &Options, _: &Options, _: Options) -> Result<(), Error> {
        Err(gap())
    }
    fn is_set_item_target(&mut self, _: &Options) -> Result<bool, Error> {
        Err(gap())
    }
    fn type_repr(&mut self, _: &Options) -> Result<String, Error> {
        Err(gap())
    }
    fn empty_set(&mut self) -> Result<Options, Error> {
        Err(gap())
    }
    fn unicode(&mut self, _: &[u8]) -> Result<Options, Error> {
        Err(gap())
    }
    fn short_string(&mut self, _: &[u8], _: &str) -> Result<Options, Error> {
        Err(gap())
    }
    fn long_integer(&mut self, _: &[u8]) -> Result<Options, Error> {
        Err(gap())
    }
    fn persistent_load(&mut self, _: &Options) -> Result<Options, Error> {
        Err(gap())
    }
    fn is_exact_integer(&mut self, _: &Options) -> Result<bool, Error> {
        Err(gap())
    }
    fn persistent_tag_is_storage(&mut self, _: &Options) -> Result<bool, Error> {
        Err(gap())
    }
    fn warning(&mut self, _: &str) -> Result<(), Error> {
        Err(gap())
    }
}
#[test]
fn original_build_restores_state_with_exact_special_class_precedence_setstate_reaccess_subclasses_slot_iteration_and_partial_mutations()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weights-build.json")).unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 190);
    for (index, c) in cases.iter().enumerate() {
        let instance = token("instance");
        let state = decode(&c["state"]);
        let world = World {
            mode: c["mode"].as_str().unwrap(),
            events: vec![],
            class: token("class"),
            instance: instance.clone(),
            other: token("other"),
            attrs: Options::dictionary(vec![]),
            reads: 0,
        };
        let mut bindings = ScalarBindings::new(world, HashState::from_seed(0));
        let mut vm = Unpickler::new("bytes");
        vm.memo.insert(0, instance.clone());
        vm.memo.insert(1, state.clone());
        let mut stream = ByteStream::new(b"h\x00h\x01b.");
        match vm.load(&mut stream, &mut bindings) {
            Ok(result) => {
                assert!(c["error"].is_null(), "case {index}");
                assert!(result.same_object(&instance));
                assert!(c["output_same"].as_bool().unwrap());
            }
            Err(e) => {
                let Error::ReferenceException { class, .. } = e.root() else {
                    panic!("case {index}: {e:?}")
                };
                assert_eq!(
                    json!({"type":class,"message":e.python_message().unwrap()}),
                    c["error"],
                    "case {index}"
                );
                assert!(!c["output_same"].as_bool().unwrap());
            }
        }
        assert_eq!(json!(bindings.inner.events), c["events"], "case {index}");
        assert_eq!(
            encode(bindings.inner.attrs.snapshot().unwrap()),
            c["attrs"],
            "case {index}"
        );
        let Options::List(stack) = &vm.stack else {
            unreachable!()
        };
        assert_eq!(
            stack.borrow().len() == 1 && stack.borrow()[0].same_object(&instance),
            c["stack_instance_same"].as_bool().unwrap(),
            "case {index}"
        );
        assert!(vm.memo.get(&1).unwrap().same_object(&state));
        assert_eq!(stream.position, c["position"].as_u64().unwrap() as usize);
    }
}
