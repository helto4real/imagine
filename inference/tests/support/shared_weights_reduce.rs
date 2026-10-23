use imagine_inference::{
    checkpoint::{
        builtin_keys::BuiltinKeys,
        builtin_scalar_ops::{self, ComplexClass},
        restricted_bindings::RestrictedBindings,
        scalar_bindings::ScalarBindings,
        weights_only::{Bindings as VmBindings, ByteStream, Unpickler},
        weights_only_globals, weights_only_newobj,
        weights_only_reduce::{self as policy, Bindings},
    },
    native::Error,
    patching::options::{GeneralMapping, Options, SourceNone, SourceNumber, SourceText},
    python_hash::HashState,
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};

struct Token(&'static str);
struct World<'a> {
    mode: &'a str,
    events: Vec<Value>,
    allowed: Options,
    user: Options,
    function: Options,
    gate: Options,
    sparse: Option<Options>,
    self_reads: usize,
    registry: Option<(&'a str, &'a str, [usize; 2])>,
}
fn error(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn gap() -> Error {
    Error::BindingGap {
        backend: "reduce_fixture".into(),
        operation: "not admitted by these own byte streams".into(),
    }
}
fn token(label: &'static str) -> Options {
    Options::Object(Rc::new(Token(label)))
}
impl World<'_> {
    fn registry_get(&mut self, kind: &'static str) -> Result<Options, Error> {
        let (mode, path, counts) = self.registry.as_mut().unwrap();
        let index = usize::from(kind == "user");
        counts[index] += 1;
        let n = counts[index];
        self.events.push(json!([kind, n]));
        if *mode == "reduce-get-error" && kind == "allowed" && n == 3 {
            return Err(error("RuntimeError", "own reduce getter failure"));
        }
        let values = GeneralMapping::empty(Rc::new(BuiltinKeys {
            hash_state: HashState::from_seed(0),
            decimal_digit_limit: Rc::new(Cell::new(4300)),
        }));
        let present = if kind == "user" {
            *mode == "user"
        } else {
            !(*mode == "user"
                || *mode == "global-disappears" && n == 2
                || *mode == "reduce-disappears" && n >= 3)
        };
        if present {
            let key = if *mode == "reduce-alias" && kind == "allowed" && n >= 3 {
                "unrelated.alias"
            } else {
                *path
            };
            values.set(key, self.function.clone())?;
        }
        Ok(values)
    }
    fn label<'a>(&self, v: &'a Options) -> &'a str {
        let Options::Object(v) = v else {
            panic!("not token")
        };
        v.downcast_ref::<Token>().unwrap().0
    }
}
impl weights_only_globals::Bindings for World<'_> {
    fn allowed(&mut self) -> Result<Options, Error> {
        if self.registry.is_some() {
            return self.registry_get("allowed");
        }
        self.events.push(json!(["allowed"]));
        if self.mode == "allowed-error" {
            return Err(error("RuntimeError", "own allowed failure"));
        }
        Ok(self.allowed.clone())
    }
    fn user(&mut self) -> Result<Options, Error> {
        if self.registry.is_some() {
            return self.registry_get("user");
        }
        self.events.push(json!(["user"]));
        if self.mode == "user-error" {
            return Err(error("RuntimeError", "own user failure"));
        }
        Ok(self.user.clone())
    }
}
impl Bindings for World<'_> {
    fn equal(&mut self, value: &Options, function: &Options) -> Result<bool, Error> {
        assert!(value.same_object(&self.gate));
        assert!(function.same_object(&self.function));
        self.events.push(json!(["equal", "gate", "function"]));
        match self.mode {
            "equal-error" => return Err(error("ValueError", "own equality failure")),
            "equal-clear" => self.allowed.clear_dict()?,
            "equal-replace" => self.allowed.set("later", self.function.clone())?,
            "equal-append" => self.allowed.set("new", self.function.clone())?,
            _ => (),
        }
        Ok(matches!(self.mode, "allowed-equal" | "user-equal"))
    }
    fn format(&mut self, value: &Options) -> Result<String, Error> {
        if let Some(class) = ComplexClass::get(value) {
            return Ok(class.representation().into());
        }
        let label = self.label(value);
        self.events.push(json!(["format", label, ""]));
        if self.mode == "format-error" {
            return Err(error("ValueError", "own format failure"));
        }
        Ok(format!("<own {label}>"))
    }
    fn bound_self(&mut self, function: &Options) -> Result<Options, Error> {
        assert!(function.same_object(&self.function));
        if self.registry.is_some() {
            return Err(error(
                "AttributeError",
                "type object 'complex' has no attribute '__self__'",
            ));
        }
        self.self_reads += 1;
        self.events.push(json!(["self", self.self_reads]));
        if self.mode == "self-subclass" {
            return Err(error("OwnAttributeError", "own subclass absent"));
        }
        if self.mode == "self-error" {
            return Err(error("RuntimeError", "own self getter failure"));
        }
        if !matches!(self.mode, "self" | "self-changes" | "self-disappears")
            || self.mode == "self-disappears" && self.self_reads == 2
        {
            return Err(error("AttributeError", "own self absent"));
        }
        Ok(token(
            if self.mode == "self-changes" && self.self_reads == 2 {
                "second"
            } else {
                "first"
            },
        ))
    }
    fn is_attribute_error(&self, error: &Error) -> bool {
        error.is_reference("AttributeError") || error.is_reference("OwnAttributeError")
    }
    fn call(&mut self, function: &Options, args: &Options) -> Result<Options, Error> {
        assert!(function.same_object(&self.function));
        if let Some(class) = ComplexClass::get(function) {
            let Options::Tuple(args) = args else {
                return Err(gap());
            };
            return class.call_positional(args);
        }
        self.events.push(json!(["call"]));
        if matches!(self.mode, "call-error" | "parameter-error") {
            return Err(error("RuntimeError", "own call failure"));
        }
        let Options::Tuple(args) = args else {
            return Err(gap());
        };
        builtin_scalar_ops::complex(args)
    }
    fn tensor_class_contains(&mut self, function: &Options) -> Result<bool, Error> {
        assert!(function.same_object(&self.function));
        if self.registry.is_some() {
            return Ok(false);
        }
        self.events.push(json!(["tensor"]));
        if self.mode == "tensor-error" {
            return Err(error("RuntimeError", "own tensor lookup failure"));
        }
        Ok(matches!(
            self.mode,
            "dense" | "sparse" | "module-error" | "append-error"
        ))
    }
    fn module_contains_sparse(&mut self, function: &Options) -> Result<bool, Error> {
        assert!(function.same_object(&self.function));
        self.events.push(json!(["module"]));
        if self.mode == "module-error" {
            return Err(error("ValueError", "own module failure"));
        }
        Ok(matches!(self.mode, "sparse" | "append-error"))
    }
    fn append_sparse(&mut self, result: &Options) -> Result<(), Error> {
        self.events.push(json!(["append"]));
        if self.mode == "append-error" {
            return Err(error("RuntimeError", "own sparse append failure"));
        }
        self.sparse = Some(result.clone());
        Ok(())
    }
}
impl weights_only_newobj::Bindings for World<'_> {
    fn parameter_class(&mut self) -> Result<Options, Error> {
        Ok(if matches!(self.mode, "parameter" | "parameter-error") {
            self.function.clone()
        } else {
            token("parameter-class")
        })
    }
    fn call_parameter(&mut self, args: &Options) -> Result<Options, Error> {
        let function = self.function.clone();
        self.call(&function, args)
    }
    fn call_new(&mut self, class: &Options, args: &Options) -> Result<Options, Error> {
        assert!(class.same_object(&self.function));
        if let Some(class) = ComplexClass::get(class) {
            let Options::Tuple(args) = args else {
                return Err(gap());
            };
            return class.call_positional(args);
        }
        self.events.push(json!(["new-get"]));
        if self.mode == "new-get-error" {
            return Err(error("ValueError", "own new getter failure"));
        }
        self.events.push(json!(["new-call"]));
        if self.mode == "new-error" {
            return Err(error("RuntimeError", "own new failure"));
        }
        let Options::Tuple(args) = args else {
            return Err(gap());
        };
        builtin_scalar_ops::complex(args)
    }
}
impl VmBindings for World<'_> {
    fn global(&mut self, _: &[u8], _: &[u8]) -> Result<Options, Error> {
        Err(gap())
    }
    fn new_object(&mut self, class: &Options, args: &Options) -> Result<Options, Error> {
        weights_only_newobj::new_object(class, args, self)
    }
    fn reduce(&mut self, function: &Options, args: &Options) -> Result<Options, Error> {
        policy::reduce(function, args, self)
    }
    fn build(&mut self, _: &Options, _: &Options) -> Result<(), Error> {
        Err(gap())
    }
    fn set_item(&mut self, _: &Options, _: &Options, _: Options) -> Result<(), Error> {
        Err(gap())
    }
    fn is_set_item_target(&mut self, _: &Options) -> Result<bool, Error> {
        Ok(false)
    }
    fn type_repr(&mut self, _: &Options) -> Result<String, Error> {
        Err(gap())
    }
    fn empty_set(&mut self) -> Result<Options, Error> {
        Err(gap())
    }
    fn unicode(&mut self, bytes: &[u8]) -> Result<Options, Error> {
        imagine_inference::python_unicode_codec::utf8(bytes, true)
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
fn encode(v: &Options, function: &Options) -> Value {
    if v.same_object(function) {
        return json!(if ComplexClass::get(function).is_some() {
            "complex-class"
        } else {
            "function"
        });
    }
    let (r, i) = SourceNumber::get(v).unwrap().complex_bits().unwrap();
    let hex = |bits: u64| {
        bits.to_le_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    json!({"real":hex(r), "imaginary":hex(i)})
}

#[test]
fn concrete_complex_global_reduce_binding_preserves_class_identity_alias_authorization_live_registry_changes_and_original_stream_failures()
 {
    check_registry("reduce");
}
#[test]
fn concrete_complex_global_newobj_binding_preserves_user_before_default_authorization_class_new_and_stream_errors()
 {
    check_registry("newobj");
}
fn check_registry(operation: &str) {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weights-reduce.json")).unwrap();
    let cases = f["registry_cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["operation"] == operation)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 49);
    let function = ComplexClass::value();
    assert!(function.same_object(&ComplexClass::value()));
    assert_eq!(
        ComplexClass::get(&function).unwrap().representation(),
        "<class 'complex'>"
    );
    assert!(function.snapshot().is_err());
    for (index, c) in cases.iter().enumerate() {
        let world = World {
            mode: "actual",
            events: vec![],
            allowed: Options::dictionary(vec![]),
            user: Options::dictionary(vec![]),
            function: function.clone(),
            gate: token("gate"),
            sparse: None,
            self_reads: 0,
            registry: Some((
                c["mode"].as_str().unwrap(),
                c["path"].as_str().unwrap(),
                [0; 2],
            )),
        };
        let mut bindings =
            ScalarBindings::new(RestrictedBindings { inner: world }, HashState::from_seed(0));
        let data = super::exact_hex(&c["bytes"]);
        let mut stream = ByteStream::new(&data);
        let mut vm = Unpickler::new("bytes");
        match vm.load(&mut stream, &mut bindings) {
            Ok(value) => {
                assert!(c["error"].is_null(), "case {index}");
                assert_eq!(encode(&value, &function), c["output"], "case {index}");
            }
            Err(error) => {
                let class = if error.is_reference("KeyError") {
                    "KeyError"
                } else {
                    match error.root() {
                        Error::ReferenceException { class, .. } => *class,
                        _ => panic!("case {index}: {error:?}"),
                    }
                };
                assert_eq!(
                    json!({"type":class,"message":error.python_message().unwrap()}),
                    c["error"],
                    "case {index}"
                );
            }
        }
        let Options::List(stack) = &vm.stack else {
            unreachable!()
        };
        assert_eq!(
            json!(
                stack
                    .borrow()
                    .iter()
                    .map(|v| encode(v, &function))
                    .collect::<Vec<_>>()
            ),
            c["stack"],
            "case {index}"
        );
        assert_eq!(
            json!(bindings.inner.inner.events),
            c["events"],
            "case {index}"
        );
        assert_eq!(
            stream.position,
            c["position"].as_u64().unwrap() as usize,
            "case {index}"
        );
    }
}
#[test]
fn original_reduce_live_values_authorization_retains_rich_comparison_mutations_argument_pop_bound_self_reaccess_sparse_registration_and_failure_prefixes()
 {
    check_operation("reduce", 168);
}
#[test]
fn original_newobj_parameter_identity_user_before_default_values_new_lookup_sparse_registration_and_failure_prefixes()
 {
    check_operation("newobj", 196);
}
fn check_operation(operation: &str, count: usize) {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weights-reduce.json")).unwrap();
    let cases = f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["operation"] == operation)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), count);
    for (index, c) in cases.iter().enumerate() {
        let mode = c["mode"].as_str().unwrap();
        let function = token("function");
        let gate = token("gate");
        let denied = matches!(
            mode,
            "missing"
                | "format-error"
                | "self"
                | "self-changes"
                | "self-disappears"
                | "self-error"
                | "self-subclass"
                | "user"
                | "user-equal"
                | "user-error"
        );
        let allowed = Options::dictionary(if denied {
            vec![]
        } else {
            vec![("first".into(), function.clone())]
        });
        if matches!(
            mode,
            "allowed-equal"
                | "equal-false"
                | "equal-error"
                | "equal-clear"
                | "equal-replace"
                | "equal-append"
        ) {
            allowed.set("first", gate.clone()).unwrap();
            if mode == "equal-replace" {
                allowed.set("later", gate.clone()).unwrap();
            }
        }
        let user = Options::dictionary(match mode {
            "user" => vec![("first".into(), function.clone())],
            "user-equal" => vec![("first".into(), gate.clone())],
            _ => vec![],
        });
        let args = Options::Tuple(Rc::new(
            c["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| match v["kind"].as_str().unwrap() {
                    "none" => SourceNone::value(),
                    "int" => SourceNumber::from_i64(v["value"].as_i64().unwrap()),
                    "str" => SourceText::from_utf8(v["value"].as_str().unwrap()),
                    _ => panic!("fixture type"),
                })
                .collect(),
        ));
        let world = World {
            mode,
            events: vec![],
            allowed,
            user,
            function: function.clone(),
            gate,
            sparse: None,
            self_reads: 0,
            registry: None,
        };
        let mut bindings = ScalarBindings::new(world, HashState::from_seed(0));
        let mut vm = Unpickler::new("bytes");
        vm.memo.insert(0, function.clone());
        vm.memo.insert(1, args.clone());
        let mut stream = ByteStream::new(if operation == "reduce" {
            b"h\x00h\x01R."
        } else {
            b"h\x00h\x01\x81."
        });
        let output = vm.load(&mut stream, &mut bindings);
        let result = match output {
            Ok(value) => {
                assert!(c["error"].is_null(), "case {index}");
                assert_eq!(encode(&value, &function), c["output"], "case {index}");
                Some(value)
            }
            Err(error) => {
                assert_eq!(
                    json!({"type":match error.root() { Error::ReferenceException {class,..} => *class, _ => panic!("case {index}: {error:?}") }, "message":error.python_message().unwrap()}),
                    c["error"],
                    "case {index}"
                );
                None
            }
        };
        let Options::List(stack) = &vm.stack else {
            unreachable!()
        };
        assert_eq!(
            json!(
                stack
                    .borrow()
                    .iter()
                    .map(|v| encode(v, &function))
                    .collect::<Vec<_>>()
            ),
            c["stack"],
            "case {index}"
        );
        assert_eq!(json!(bindings.inner.events), c["events"], "case {index}");
        assert_eq!(stream.position, c["position"].as_u64().unwrap() as usize);
        assert!(vm.memo.get(&0).unwrap().same_object(&function));
        assert!(vm.memo.get(&1).unwrap().same_object(&args));
        assert_eq!(
            bindings
                .inner
                .sparse
                .as_ref()
                .zip(result.as_ref())
                .is_some_and(|(a, b)| a.same_object(b)),
            c["sparse_result_same"].as_bool().unwrap()
        );
    }
}
