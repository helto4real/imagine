use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::inpaint_state::{self, Bindings, InpaintClass, InpaintState},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
struct Number {
    name: &'static str,
    value: f64,
}
fn number(name: &'static str, value: f64) -> Options {
    Options::Object(Rc::new(Number { name, value }))
}
fn n(value: &Options) -> &Number {
    let Options::Object(value) = value else {
        panic!("own number")
    };
    value.downcast_ref().unwrap()
}
fn role(value: &str) -> Options {
    Options::Atom(V::Text(value.into()))
}
fn text(value: &Options) -> &str {
    let Options::Atom(V::Text(value)) = value else {
        panic!("own role")
    };
    value
}
fn options(value: &Options) -> Value {
    match value {
        Options::Atom(V::None) => Value::Null,
        Options::Dict(v) => Value::Object(
            v.borrow()
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        match v {
                            Options::Atom(V::Bool(v)) => json!(v),
                            _ => json!(text(v)),
                        },
                    )
                })
                .collect(),
        ),
        _ => panic!("own options"),
    }
}
struct Data<'a> {
    case: &'a Value,
    state: &'a InpaintState,
    events: RefCell<Vec<Value>>,
}
impl Data<'_> {
    fn emit(&self, value: Value) -> Result<(), Error> {
        let index = self.events.borrow().len();
        self.events.borrow_mut().push(value);
        if self.case["failure"].as_u64() == Some(index as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own inpaint failure".into(),
            });
        }
        Ok(())
    }
}
impl Bindings for Data<'_> {
    fn contains(&self, object: &Options, key: &str) -> Result<bool, Error> {
        if matches!(object, Options::Atom(V::None)) {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "argument of type 'NoneType' is not iterable".into(),
            });
        }
        Ok(object.get(key)?.is_some())
    }
    fn item(&self, object: &Options, key: &str) -> Result<Options, Error> {
        object.item(&role(key))
    }
    fn attribute(&self, object: &Options, name: &str) -> Result<Options, Error> {
        self.emit(json!(["attribute", text(object), name]))?;
        Ok(role(&if name == "inner_model" {
            format!("inner-{}", text(object))
        } else {
            format!("scale:{}", text(object))
        }))
    }
    fn call(
        &self,
        function: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
    ) -> Result<Options, Error> {
        let function = text(function);
        if function == "mask_fn" {
            assert_eq!(args.len(), 2);
            assert_eq!(kwargs.len(), 1);
            assert_eq!(kwargs[0].0, "extra_options");
            let extra = &kwargs[0].1;
            let Options::Dict(entries) = extra else {
                panic!("dictionary")
            };
            assert_eq!(
                entries
                    .borrow()
                    .iter()
                    .map(|(key, _)| key.as_str())
                    .collect::<Vec<_>>(),
                vec!["model", "sigmas"]
            );
            self.emit(json!([
                "mask-function",
                n(&args[0]).value,
                n(&args[1]).value,
                text(&extra.get("model")?.unwrap()),
                text(&extra.get("sigmas")?.unwrap())
            ]))?;
            if self.case["mode"] == "replace-mask-model" {
                self.state.fields.set("inner_model", role("model1"));
                self.state.fields.set("sigmas", role("sigmas1"));
            }
            return Ok(number("new-mask", n(&args[1]).value * 0.5));
        }
        if let Some(owner) = function.strip_prefix("scale:") {
            assert!(args.is_empty());
            assert_eq!(
                kwargs
                    .iter()
                    .map(|(key, _)| key.as_str())
                    .collect::<Vec<_>>(),
                vec!["x", "sigma", "noise", "latent_image", "denoise_mask"]
            );
            let get = |name: &str| &kwargs.iter().find(|(key, _)| key == name).unwrap().1;
            let encoded: serde_json::Map<String, Value> = kwargs
                .iter()
                .map(|(key, value)| (key.clone(), json!(n(value).value)))
                .collect();
            self.emit(json!(["scale", owner, encoded]))?;
            if self.case["mode"] == "replace-inner" {
                self.state.fields.set("inner_model", role("model1"));
            }
            return Ok(number(
                "scaled",
                n(get("latent_image")).value * 0.75 + n(get("noise")).value * 0.25,
            ));
        }
        assert_eq!(args.len(), 2);
        assert_eq!(
            kwargs
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>(),
            vec!["model_options", "seed"]
        );
        let model_options = &kwargs[0].1;
        let Options::Atom(V::Int(seed)) = &kwargs[1].1 else {
            panic!("seed")
        };
        self.emit(json!([
            "model",
            function,
            n(&args[0]).value,
            n(&args[1]).value,
            options(model_options),
            seed
        ]))?;
        if self.case["mode"] == "replace-latent" {
            self.state
                .fields
                .set("latent_image", number("replacement-latent", 20.));
        }
        if self.case["options_mode"] == "omitted" {
            model_options.set("seen", Options::Atom(V::Bool(true)))?;
        }
        Ok(number(
            "denoised",
            n(&args[0]).value * 0.5 + n(&args[1]).value,
        ))
    }
    fn subtract_from_one(&self, value: &Options) -> Result<Options, Error> {
        self.emit(json!(["subtract", n(value).value]))?;
        Ok(number("inverse-mask", 1. - n(value).value))
    }
    fn multiply(&self, left: &Options, right: &Options) -> Result<Options, Error> {
        self.emit(json!(["multiply", n(left).value, n(right).value]))?;
        if n(left).name == "x" && self.case["mode"] == "replace-noise" {
            self.state
                .fields
                .set("noise", number("replacement-noise", 9.));
        }
        Ok(number("product", n(left).value * n(right).value))
    }
    fn add(&self, left: &Options, right: &Options) -> Result<Options, Error> {
        self.emit(json!(["add", n(left).value, n(right).value]))?;
        Ok(number("sum", n(left).value + n(right).value))
    }
}
#[test]
fn original_mutable_inpaint_preserves_mask_model_noise_latent_rebinding_default_dictionary_and_original_failure_prefixes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-inpaint-state.json")).unwrap();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let class = InpaintClass::default();
        let state = InpaintState::new(role("model0"), role("sigmas0"));
        state.fields.set("noise", number("noise", 2.));
        state.fields.set("latent_image", number("latent", 10.));
        let binding = Data {
            case,
            state: &state,
            events: RefCell::new(vec![]),
        };
        let explicit = Options::dictionary(if case["function"] == true {
            vec![("denoise_mask_function".into(), role("mask_fn"))]
        } else {
            vec![]
        });
        let model_options = match case["options_mode"].as_str().unwrap() {
            "omitted" => None,
            "none" => Some(Options::Atom(V::None)),
            _ => Some(explicit),
        };
        let mut values = vec![];
        let result = (|| -> Result<(), Error> {
            for _ in 0..if case["options_mode"] == "omitted" {
                2
            } else {
                1
            } {
                let value = class.call(
                    &state,
                    inpaint_state::Arguments {
                        x: number("x", 8.),
                        sigma: number("sigma", 0.75),
                        mask: if case["masked"] == true {
                            number("mask", 0.25)
                        } else {
                            Options::Atom(V::None)
                        },
                        model_options: model_options.clone(),
                        seed: Options::Atom(V::Int(901)),
                    },
                    &binding,
                )?;
                values.push(n(&value).value);
            }
            Ok(())
        })();
        let (error, output) = match result {
            Ok(()) => (Value::Null, json!(values)),
            Err(Error::ReferenceException { class, message }) => {
                (json!([class, message]), Value::Null)
            }
            Err(error) => panic!("unbound inpaint {error:?}"),
        };
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(output, case["output"], "output {index}");
        assert_eq!(
            json!(*binding.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(
            json!({"noise":n(&state.fields.get("noise").unwrap()).value,"latent":n(&state.fields.get("latent_image").unwrap()).value,"model":text(&state.fields.get("inner_model").unwrap()),"sigmas":text(&state.fields.get("sigmas").unwrap()),"defaults":options(&class.default_options())}),
            case["final"],
            "final {index}"
        );
    }
}
