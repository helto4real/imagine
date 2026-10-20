use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::{dpm_wrapper::Bindings, factory::DpmWrapper},
};
use serde_json::{Value, json};
use std::cell::RefCell;
fn value(v: &Value) -> Options {
    Options::from_value(match v {
        Value::Null => V::None,
        Value::Bool(v) => V::Bool(*v),
        Value::Number(v) if v.is_i64() => V::Int(v.as_i64().unwrap()),
        Value::Number(v) => V::Float(v.as_f64().unwrap()),
        Value::String(v) => V::Text(v.clone()),
        _ => panic!("own scalar"),
    })
}
fn text(s: &str) -> Options {
    Options::Atom(V::Text(s.into()))
}
fn encode(v: &Options) -> Value {
    match v {
        Options::Atom(V::Text(v)) => json!(v),
        Options::Atom(V::Int(v)) => json!(v),
        Options::Atom(V::Float(v)) => json!(v),
        Options::Atom(V::Bool(v)) => json!(v),
        Options::Atom(V::None) => Value::Null,
        _ => panic!("own token"),
    }
}
struct Data<'a> {
    case: &'a Value,
    events: RefCell<Vec<Value>>,
}
impl Data<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let index = self.events.borrow().len();
        self.events.borrow_mut().push(event);
        if self.case["failure"].as_u64() == Some(index as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own DPM call failure".into(),
            });
        }
        Ok(())
    }
}
impl Bindings for Data<'_> {
    fn length(&self, _: &Options) -> Result<i64, Error> {
        self.emit(json!(["length"]))?;
        match self.case["shape"].as_str().unwrap() {
            "scalar" => Err(Error::ReferenceException {
                class: "TypeError",
                message: "own scalar length".into(),
            }),
            "empty" => Ok(0),
            "one" => Ok(1),
            _ => Ok(4),
        }
    }
    fn index(&self, _: &Options, index: i64) -> Result<Options, Error> {
        self.emit(json!(["index", index]))?;
        Ok(text(match index {
            -1 => "last",
            -2 => "penultimate",
            0 => "first",
            _ => panic!("index"),
        }))
    }
    fn equals_zero(&self, sigma: &Options) -> Result<bool, Error> {
        self.emit(json!(["equals-zero", encode(sigma), 0]))?;
        if self.case["shape"] == "ambiguous" {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: "own ambiguous truth".into(),
            });
        }
        Ok(self.case["terminal"] == true && encode(sigma) == "last")
    }
    fn module_attribute(&self, name: &str) -> Result<Options, Error> {
        self.emit(json!(["module", name]))?;
        Ok(text(name))
    }
    fn call(
        &self,
        _: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
    ) -> Result<Options, Error> {
        self.emit(json!([
            "solve",
            args.iter().map(encode).collect::<Vec<_>>(),
            kwargs
                .iter()
                .map(|(k, v)| json!([k, encode(v)]))
                .collect::<Vec<_>>()
        ]))?;
        Ok(text("result"))
    }
}
#[test]
fn original_nested_dpm_calls_preserve_keyword_binding_advanced_option_forwarding_early_aliases_module_lookup_and_failure_prefixes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dpm-call.json")).unwrap();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let binding = Data {
            case,
            events: RefCell::new(vec![]),
        };
        let mut args: Vec<_> = ["model", "noise", "sigmas", "extra", "callback", "disable"]
            .into_iter()
            .map(text)
            .collect();
        let mut kwargs = vec![];
        let style = case["style"].as_str().unwrap();
        match style {
            "keywords" => {
                kwargs = [
                    "model",
                    "noise",
                    "sigmas",
                    "extra_args",
                    "callback",
                    "disable",
                ]
                .into_iter()
                .zip(args.drain(..))
                .map(|(k, v)| (k.into(), v))
                .collect()
            }
            "options" => {
                kwargs = vec![
                    ("eta".into(), value(&json!(0.5))),
                    ("s_noise".into(), value(&json!(1.25))),
                    ("return_info".into(), value(&json!(true))),
                    ("noise_sampler".into(), value(&Value::Null)),
                    ("own".into(), value(&json!(7))),
                ]
            }
            "duplicate" => kwargs.push(("model".into(), text("another"))),
            "unexpected" => kwargs.push(("own".into(), value(&json!(7)))),
            _ => {
                if let Some(count) = style.strip_prefix("arity:") {
                    let count: usize = count.parse().unwrap();
                    args.truncate(count);
                    args.resize_with(count, || text("overflow"));
                }
            }
        }
        let wrapper = if case["name"] == "dpm_fast" {
            DpmWrapper::Fast
        } else {
            DpmWrapper::Adaptive
        };
        let (output, error) = match wrapper.call(&args, &kwargs, &binding) {
            Ok(v) => (encode(&v), Value::Null),
            Err(Error::ReferenceException { class, message }) => {
                (Value::Null, json!([class, message]))
            }
            Err(e) => panic!("unbound {e:?}"),
        };
        assert_eq!(output, case["output"], "output {index}");
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(
            json!(*binding.events.borrow()),
            case["events"],
            "events {index}"
        );
    }
}
