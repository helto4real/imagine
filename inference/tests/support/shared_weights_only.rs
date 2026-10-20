use imagine_inference::{
    checkpoint::weights_only::{Bindings, ByteStream, Unpickler},
    execution_value::Value as V,
    native::Error,
    patching::options::{Options, SourceText},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, rc::Rc};
enum Primitive {
    Integer(Vec<u8>),
    Bytes(Vec<u8>),
    Set,
}
fn primitive(value: &Options) -> Option<&Primitive> {
    let Options::Object(value) = value else {
        return None;
    };
    value.downcast_ref()
}
fn failure(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn unpickling(message: impl Into<String>) -> Error {
    failure("UnpicklingError", message)
}
fn text(value: &Options) -> Option<String> {
    if let Some(value) = SourceText::get(value) {
        return value.utf8();
    }
    if let Options::Atom(V::Text(value)) = value {
        return Some(value.clone());
    }
    None
}
fn category(value: &Options) -> &'static str {
    if SourceText::get(value).is_some() {
        return "str";
    }
    if let Some(value) = primitive(value) {
        return match value {
            Primitive::Integer(_) => "int",
            Primitive::Bytes(_) => "bytes",
            Primitive::Set => "set",
        };
    }
    match value {
        Options::Atom(V::None) => "NoneType",
        Options::Atom(V::Bool(_)) => "bool",
        Options::Atom(V::Int(_) | V::UInt(_)) => "int",
        Options::Atom(V::Float(_)) => "float",
        Options::Atom(V::Text(_)) => "str",
        Options::List(_) => "list",
        Options::Dict(_) => "dict",
        Options::Tuple(_) => "tuple",
        _ => panic!("unexpected primitive fixture kind"),
    }
}
fn scalar_repr(value: &Options) -> String {
    match value {
        Options::Atom(V::Int(v)) => v.to_string(),
        _ => panic!("reference only calls rejected primitive reduce/newobj with int"),
    }
}
#[derive(Default)]
struct World {
    warnings: Vec<Value>,
}
impl Bindings for World {
    fn global(&mut self, module: &[u8], name: &[u8]) -> Result<Options, Error> {
        imagine_inference::checkpoint::weights_only_globals::resolve(
            std::str::from_utf8(module).unwrap(),
            std::str::from_utf8(name).unwrap(),
            self,
        )
    }
    fn new_object(&mut self, class: &Options, _args: &Options) -> Result<Options, Error> {
        Err(unpickling(format!(
            "Can only create new object for nn.Parameter or classes allowlisted via `add_safe_globals` but got {}",
            scalar_repr(class)
        )))
    }
    fn reduce(&mut self, function: &Options, _args: &Options) -> Result<Options, Error> {
        Err(unpickling(format!(
            "Trying to call reduce for unrecognized function {}",
            scalar_repr(function)
        )))
    }
    fn build(&mut self, instance: &Options, _state: &Options) -> Result<(), Error> {
        Err(unpickling(format!(
            "Can only build Tensor, Parameter, OrderedDict or types allowlisted via `add_safe_globals`, but got {}",
            self.type_repr(instance)?
        )))
    }
    fn set_item(&mut self, target: &Options, key: &Options, value: Options) -> Result<(), Error> {
        let key = text(key).expect("retained positive dictionary keys are text");
        target.set(&key, value)
    }
    fn is_set_item_target(&mut self, target: &Options) -> Result<bool, Error> {
        Ok(matches!(target, Options::Dict(_)))
    }
    fn type_repr(&mut self, value: &Options) -> Result<String, Error> {
        Ok(format!("<class '{}'>", category(value)))
    }
    fn empty_set(&mut self) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(Primitive::Set)))
    }
    fn unicode(&mut self, bytes: &[u8]) -> Result<Options, Error> {
        imagine_inference::python_unicode_codec::utf8(bytes, true)
    }
    fn short_string(&mut self, bytes: &[u8], encoding: &str) -> Result<Options, Error> {
        match encoding {
            "bytes" => Ok(Options::Object(Rc::new(Primitive::Bytes(bytes.into())))),
            "ASCII" => imagine_inference::python_unicode_codec::ascii(bytes),
            "utf-8" => imagine_inference::python_unicode_codec::utf8(bytes, false),
            _ => panic!("unexpected fixture encoding"),
        }
    }
    fn long_integer(&mut self, bytes: &[u8]) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(Primitive::Integer(bytes.into()))))
    }
    fn persistent_load(&mut self, _pid: &Options) -> Result<Options, Error> {
        Err(unpickling("unsupported persistent id encountered"))
    }
    fn is_exact_integer(&mut self, value: &Options) -> Result<bool, Error> {
        Ok(matches!(value, Options::Atom(V::Int(_) | V::UInt(_)))
            || matches!(primitive(value), Some(Primitive::Integer(_))))
    }
    fn persistent_tag_is_storage(&mut self, value: &Options) -> Result<bool, Error> {
        Ok(text(value).is_some_and(|v| v == "storage")
            || matches!(primitive(value),Some(Primitive::Bytes(v))if v==b"storage"))
    }
    fn warning(&mut self, message: &str) -> Result<(), Error> {
        self.warnings
            .push(json!({"category":"UserWarning","message":message}));
        Ok(())
    }
}
impl imagine_inference::checkpoint::weights_only_globals::Bindings for World {
    fn allowed(&mut self) -> Result<Options, Error> {
        Ok(Options::dictionary(vec![]))
    }
    fn user(&mut self) -> Result<Options, Error> {
        Ok(Options::dictionary(vec![]))
    }
}
fn integer(mut bytes: Vec<u8>) -> Value {
    if bytes.is_empty() {
        bytes.push(0);
    }
    while bytes.len() > 1 {
        let n = bytes.len();
        if (bytes[n - 1] == 0 && bytes[n - 2] < 128) || (bytes[n - 1] == 255 && bytes[n - 2] >= 128)
        {
            bytes.pop();
        } else {
            break;
        }
    }
    json!({"type":"int","bytes":bytes.iter().map(|b|format!("{b:02x}")).collect::<String>()})
}
#[derive(Default)]
struct Graph {
    ids: BTreeMap<(u8, usize), usize>,
    nodes: Vec<Value>,
    live: Vec<Options>,
}
impl Graph {
    fn value(&mut self, value: &Options) -> Value {
        if let Some(value) = SourceText::get(value) {
            return json!({"type":"str","points":value.codepoints()});
        }
        if let Some(value) = primitive(value) {
            match value {
                Primitive::Integer(v) => return integer(v.clone()),
                Primitive::Bytes(v) => {
                    return json!({"type":"bytes","bytes":v.iter().map(|b|format!("{b:02x}")).collect::<String>()});
                }
                Primitive::Set => {}
            }
        }
        if let Options::Atom(value) = value {
            return match value {
                V::None => json!({"type":"none"}),
                V::Bool(value) => json!({"type":"bool","value":value}),
                V::Int(value) => integer(value.to_le_bytes().to_vec()),
                V::UInt(value) => {
                    let mut bytes = value.to_le_bytes().to_vec();
                    bytes.push(0);
                    integer(bytes)
                }
                V::Float(value) => {
                    json!({"type":"float","bits":format!("{:016x}",value.to_bits())})
                }
                V::Text(value) => {
                    json!({"type":"str","points":value.chars().map(u32::from).collect::<Vec<_>>()})
                }
                _ => panic!("unexpected primitive atom"),
            };
        }
        let key = match value {
            Options::List(v) => (0, Rc::as_ptr(v) as usize),
            Options::Tuple(v) => (1, Rc::as_ptr(v) as usize),
            Options::Dict(v) => (2, Rc::as_ptr(v) as usize),
            Options::Object(v) => (3, Rc::as_ptr(v).cast::<()>() as usize),
            _ => unreachable!(),
        };
        if let Some(index) = self.ids.get(&key) {
            return json!({"ref":index});
        }
        let index = self.nodes.len();
        self.ids.insert(key, index);
        self.nodes.push(Value::Null);
        self.live.push(value.clone());
        let node = match value {
            Options::List(values) => {
                let values = values.borrow().clone();
                json!({"type":"list","items":values.iter().map(|v|self.value(v)).collect::<Vec<_>>()})
            }
            Options::Tuple(values) => {
                json!({"type":"tuple","items":values.iter().map(|v|self.value(v)).collect::<Vec<_>>()})
            }
            Options::Dict(values) => {
                let values = values
                    .borrow()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>();
                json!({"type":"dict","items":values.iter().map(|(k,v)|json!([self.value(&Options::Atom(V::Text(k.clone()))),self.value(v)])).collect::<Vec<_>>()})
            }
            Options::Object(_) => {
                assert!(matches!(primitive(value), Some(Primitive::Set)));
                json!({"type":"set","items":[]})
            }
            _ => unreachable!(),
        };
        self.nodes[index] = node;
        json!({"ref":index})
    }
    fn cleanup(&self) {
        // Fixture teardown breaks its deliberately created cycles. This is
        // not evidence that the runtime cycle collector is complete.
        for value in &self.live {
            match value {
                Options::List(values) => {
                    let old = std::mem::take(&mut *values.borrow_mut());
                    drop(old);
                }
                Options::Dict(_) => {
                    value.clear_dict().unwrap();
                }
                _ => {}
            }
        }
    }
}
#[test]
fn original_restricted_unpickler_byte_machine_preserves_primitives_memo_cycles_stack_mark_partial_errors_warnings_positions_and_repeated_loads()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-weights-only.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 270);
    for case in fixture["cases"].as_array().unwrap() {
        let bytes = super::exact_hex(&case["bytes"]);
        let mut input = ByteStream::new(&bytes);
        let mut machine = Unpickler::new(case["encoding"].as_str().unwrap());
        let mut world = World::default();
        let mut graphs = vec![];
        for expected in case["runs"].as_array().unwrap() {
            world.warnings.clear();
            let result = machine.load(&mut input, &mut world);
            let mut graph = Graph::default();
            let mut record = if expected["error"].is_string() {
                let error = match result {
                    Err(error) => error,
                    Ok(_) => panic!("expected source error: {}", case["label"]),
                };
                super::binding_tests::reference_error(error, expected);
                json!({"error":expected["error"],"message":expected["message"]})
            } else {
                json!({"output":graph.value(&result.unwrap_or_else(|error|panic!("{}: {error:?}",case["label"])))})
            };
            record["position"] = json!(input.position);
            record["proto"] = json!(machine.proto);
            record["stack"] = graph.value(&machine.stack);
            record["metastack"] = json!(
                machine
                    .metastack
                    .iter()
                    .map(|v| graph.value(v))
                    .collect::<Vec<_>>()
            );
            record["memo"] = json!(
                machine
                    .memo
                    .iter()
                    .map(|(key, value)| json!([key, graph.value(value)]))
                    .collect::<Vec<_>>()
            );
            record["warnings"] = json!(world.warnings);
            record["nodes"] = json!(graph.nodes);
            assert_eq!(record, *expected, "{}", case["label"]);
            graphs.push(graph);
        }
        for graph in graphs {
            graph.cleanup();
        }
    }
}
