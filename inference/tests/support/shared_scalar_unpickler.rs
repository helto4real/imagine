use imagine_inference::{
    checkpoint::{
        scalar_bindings::ScalarBindings,
        weights_only::{Bindings, ByteStream, Unpickler},
    },
    native::Error,
    patching::options::{
        GeneralMapping, GeneralSet, Options, SourceByteArray, SourceBytes, SourceNone,
        SourceNumber as N, SourceText,
    },
    python_hash::HashState,
};
use serde_json::{Value, json};
#[derive(Default)]
struct Native {
    warnings: Vec<String>,
}
fn gap() -> Error {
    Error::BindingGap {
        backend: "scalar_vm_test".into(),
        operation: "global/tensor materialization is outside this scalar fixture".into(),
    }
}
impl Bindings for Native {
    fn global(&mut self, _: &[u8], _: &[u8]) -> Result<Options, Error> {
        Err(gap())
    }
    fn new_object(&mut self, _: &Options, _: &Options) -> Result<Options, Error> {
        Err(gap())
    }
    fn reduce(&mut self, _: &Options, _: &Options) -> Result<Options, Error> {
        Err(gap())
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
    fn type_repr(&mut self, v: &Options) -> Result<String, Error> {
        Ok(format!(
            "<class '{}'>",
            match v {
                Options::List(_) => "list",
                Options::Tuple(_) => "tuple",
                _ => "object",
            }
        ))
    }
    fn empty_set(&mut self) -> Result<Options, Error> {
        Err(gap())
    }
    fn unicode(&mut self, b: &[u8]) -> Result<Options, Error> {
        imagine_inference::python_unicode_codec::utf8(b, true)
    }
    fn short_string(&mut self, b: &[u8], encoding: &str) -> Result<Options, Error> {
        assert_eq!(encoding, "utf-8");
        imagine_inference::python_unicode_codec::utf8(b, false)
    }
    fn long_integer(&mut self, _: &[u8]) -> Result<Options, Error> {
        Err(gap())
    }
    fn persistent_load(&mut self, _: &Options) -> Result<Options, Error> {
        Err(Error::ReferenceException {
            class: "UnpicklingError",
            message: "unsupported persistent id encountered".into(),
        })
    }
    fn is_exact_integer(&mut self, _: &Options) -> Result<bool, Error> {
        Ok(false)
    }
    fn persistent_tag_is_storage(&mut self, _: &Options) -> Result<bool, Error> {
        Ok(false)
    }
    fn warning(&mut self, m: &str) -> Result<(), Error> {
        self.warnings.push(m.into());
        Ok(())
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
struct Graph {
    seen: Vec<Options>,
    nodes: Vec<Value>,
    roots: serde_json::Map<String, Value>,
}
impl Graph {
    fn new() -> Self {
        Self {
            seen: vec![],
            nodes: vec![],
            roots: Default::default(),
        }
    }
    fn root(&mut self, name: &str, v: &Options) {
        let id = self.node(v);
        self.roots.insert(name.into(), json!(id));
    }
    fn node(&mut self, v: &Options) -> usize {
        if let Some(i) = self.seen.iter().position(|s| s.same_object(v)) {
            return i;
        }
        let i = self.nodes.len();
        self.nodes.push(Value::Null);
        self.seen.push(v.clone());
        let node = if let Some(bytes) = SourceBytes::get(v) {
            json!({"kind":"bytes","data":hex(bytes.bytes())})
        } else if let Some(bytes) = SourceByteArray::get(v) {
            json!({"kind":"bytearray","data":hex(&bytes.bytes())})
        } else if SourceNone::get(v).is_some() {
            json!({"kind":"NoneType"})
        } else if let Some(n) = N::get(v) {
            if n.is_exact_integer() {
                json!({"kind":"int","bytes":hex(n.integer_bytes().unwrap())})
            } else if let Some(bits) = n.float_bits() {
                json!({"kind":"float","bits":hex(&bits.to_le_bytes())})
            } else if n.complex_bits().is_some() {
                panic!("complex is outside this VM fixture")
            } else {
                json!({"kind":"bool","value":n.truth()})
            }
        } else if let Some(t) = SourceText::get(v) {
            json!({"kind":"str","points":t.codepoints()})
        } else if let Some(set) = GeneralSet::get(v) {
            let items = set
                .entries()
                .iter()
                .map(|(value, _)| self.node(value))
                .collect::<Vec<_>>();
            json!({"kind":"set","items":items})
        } else if let Some(m) = GeneralMapping::get(v) {
            let entries = m
                .dictionary
                .entries()
                .iter()
                .map(|(k, v)| json!([self.node(k), self.node(v)]))
                .collect::<Vec<_>>();
            json!({"kind":"dict","entries":entries})
        } else {
            match v {
                Options::List(v) => {
                    let items = v.borrow().iter().map(|v| self.node(v)).collect::<Vec<_>>();
                    json!({"kind":"list","items":items})
                }
                Options::Tuple(v) => {
                    let items = v.iter().map(|v| self.node(v)).collect::<Vec<_>>();
                    json!({"kind":"tuple","items":items})
                }
                _ => panic!("unexpected native graph object"),
            }
        };
        self.nodes[i] = node;
        i
    }
    fn finish(self) -> Value {
        json!({"roots":self.roots,"nodes":self.nodes})
    }
}
#[test]
fn weights_only_scalar_constructors_preserve_real_builtin_owners_memo_stacks_keys_and_original_errors()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-scalar-unpickler.json")).unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 66);
    for (index, c) in cases.iter().enumerate() {
        let bytes = super::exact_hex(&c["bytes"]);
        let mut input = ByteStream::new(&bytes);
        let mut vm = Unpickler::new(c["encoding"].as_str().unwrap());
        let mut bindings = ScalarBindings::new(Native::default(), HashState::from_seed(0));
        for expected in c["loads"].as_array().unwrap() {
            bindings.inner.warnings.clear();
            let out = vm.load(&mut input, &mut bindings);
            let mut graph = Graph::new();
            if expected["error"].is_object() {
                match out
                    .err()
                    .unwrap_or_else(|| panic!("expected error case {index}"))
                {
                    Error::ReferenceException { class, message } => {
                        assert_eq!(class, expected["error"]["type"].as_str().unwrap());
                        assert_eq!(message, expected["error"]["message"].as_str().unwrap());
                    }
                    e => panic!("case {index}: {e:?}"),
                }
            } else {
                graph.root(
                    "output",
                    &out.unwrap_or_else(|e| panic!("case {index}: {e:?}")),
                );
            }
            graph.root("stack", &vm.stack);
            for (i, value) in vm.metastack.iter().enumerate() {
                graph.root(&format!("mark{i}"), value);
            }
            for (i, value) in vm.memo.iter() {
                graph.root(&format!("memo{i}"), value);
            }
            assert_eq!(
                input.position,
                expected["position"].as_u64().unwrap() as usize,
                "stream position case {index}"
            );
            assert_eq!(
                vm.proto,
                i32::try_from(expected["proto"].as_i64().unwrap()).unwrap()
            );
            assert_eq!(json!(bindings.inner.warnings), expected["warnings"]);
            assert_eq!(
                graph.finish(),
                expected["graph"],
                "native scalar VM graph case {index}"
            );
        }
    }
}

#[test]
fn concrete_none_hash_and_decoder_singletons_match_locked_cpython_without_interning_general_allocations()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-scalar-unpickler.json")).unwrap();
    let first = SourceNone::value();
    assert!(first.same_object(&SourceNone::value()));
    assert_eq!(
        SourceNone::get(&first).unwrap().hash(),
        f["none_hash"].as_i64().unwrap()
    );
    for c in f["decoder_identity"].as_array().unwrap() {
        let bytes = super::exact_hex(&c["bytes"]);
        let a = imagine_inference::python_unicode_codec::utf8(&bytes, true).unwrap();
        let b = imagine_inference::python_unicode_codec::utf8(&bytes, true).unwrap();
        assert_eq!(a.same_object(&b), c["same"].as_bool().unwrap());
        let point = c["point"].as_u64().unwrap() as u32;
        assert_eq!(SourceText::get(&a).unwrap().codepoints(), &[point]);
        if point < 128 {
            let ascii = imagine_inference::python_unicode_codec::ascii(&[point as u8]).unwrap();
            assert_eq!(
                ascii.same_object(&a),
                c["ascii_cross_same"].as_bool().unwrap()
            );
        }
    }
    let a = SourceText::from_utf8("A");
    let b = SourceText::from_utf8("A");
    assert!(!a.same_object(&b));
}
