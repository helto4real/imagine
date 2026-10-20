use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::{
        factory::SamplerState,
        sampler_state::{self, Bindings, Callback},
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
fn role(v: &str) -> Options {
    Options::Atom(V::Text(v.into()))
}
fn integer(v: i64) -> Options {
    Options::Atom(V::Int(v))
}
fn label(v: &Options) -> &str {
    let Options::Atom(V::Text(v)) = v else {
        panic!("own role")
    };
    v
}
fn encode(v: &Options) -> Value {
    match v {
        Options::Atom(V::None) => Value::Null,
        Options::Atom(V::Text(v)) => json!(v),
        Options::Atom(V::Int(v)) => json!(v),
        Options::Atom(V::Float(v)) => json!(v),
        Options::Atom(V::Bool(v)) => json!(v),
        Options::Tuple(v) => json!(v.iter().map(encode).collect::<Vec<_>>()),
        Options::Dict(v) => Value::Object(
            v.borrow()
                .iter()
                .map(|(k, v)| (k.clone(), encode(v)))
                .collect(),
        ),
        _ => panic!("own encoded value"),
    }
}
struct Function(Callback);
struct Data {
    case: Value,
    state: Rc<SamplerState>,
    extra: Options,
    dicts: RefCell<Vec<(Options, String)>>,
    fields: RefCell<BTreeMap<(String, String), Options>>,
    model_k: Options,
    saved: RefCell<Vec<Options>>,
    events: RefCell<Vec<Value>>,
}
impl Data {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let index = self.events.borrow().len();
        self.events.borrow_mut().push(event);
        if self.case["failure"].as_u64() == Some(index as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own stateful sampler failure".into(),
            });
        }
        Ok(())
    }
    fn dict_label(&self, value: &Options) -> String {
        self.dicts
            .borrow()
            .iter()
            .find(|(v, _)| v.same_object(value))
            .map(|(_, name)| name.clone())
            .expect("tracked own dictionary")
    }
    fn step(&self, name: &str, index: i64) -> Options {
        let value = Options::dictionary(vec![
            ("i".into(), integer(index)),
            ("x".into(), role("x")),
            ("denoised".into(), role("denoised")),
            ("sigma".into(), role("sigma")),
            ("sigma_hat".into(), role("hat")),
        ]);
        self.dicts.borrow_mut().push((value.clone(), name.into()));
        value
    }
    fn invoke(&self, callback: &Options, value: &Options) -> Result<Options, Error> {
        let Options::Object(callback) = callback else {
            panic!("owned callback")
        };
        (callback.downcast_ref::<Function>().unwrap().0)(value)
    }
}
impl Bindings for Data {
    fn attribute(&self, object: &Options, name: &str) -> Result<Options, Error> {
        let owner = label(object);
        if owner == "physical" && name == "__class__" {
            return Ok(role("class"));
        }
        if owner == "class" && name == "__name__" {
            return Ok(role("Obj"));
        }
        self.emit(json!(["attribute", owner, name]))?;
        if let Some(value) = self
            .fields
            .borrow()
            .get(&(owner.into(), name.into()))
            .cloned()
        {
            return Ok(value);
        }
        if matches!(name, "to" | "noise_scaling" | "inverse_noise_scaling") {
            return Ok(role(&format!("{owner}.{name}")));
        }
        Err(Error::ReferenceException {
            class: "AttributeError",
            message: format!("own missing {owner}.{name}"),
        })
    }
    fn attribute_or(
        &self,
        object: &Options,
        name: &str,
        default: Options,
    ) -> Result<Options, Error> {
        if name == "__name__" && label(object).starts_with("sampler_") {
            Ok(object.clone())
        } else {
            Ok(default)
        }
    }
    fn get(&self, object: &Options, key: &str, default: Options) -> Result<Options, Error> {
        self.emit(json!([
            "get",
            self.dict_label(object),
            key,
            encode(&default)
        ]))?;
        Ok(object.get(key)?.unwrap_or(default))
    }
    fn item(&self, object: &Options, key: &str) -> Result<Options, Error> {
        self.emit(json!(["item", self.dict_label(object), key]))?;
        object.item(&role(key))
    }
    fn set_item(&self, object: &Options, key: &str, value: Options) -> Result<(), Error> {
        self.emit(json!(["set", self.dict_label(object), key, encode(&value)]))?;
        object.set(key, value)
    }
    fn set_attribute(&self, object: &Options, name: &str, value: Options) -> Result<(), Error> {
        assert_eq!(label(object), "model-k");
        self.emit(json!(["set-attribute", "model-k", name, encode(&value)]))?;
        self.model_k.set(name, value)
    }
    fn truth(&self, value: &Options) -> Result<bool, Error> {
        let Options::Atom(V::Bool(v)) = value else {
            panic!("own bool")
        };
        Ok(*v)
    }
    fn index(&self, object: &Options, index: i64) -> Result<Options, Error> {
        self.emit(json!(["index", label(object), index]))?;
        Ok(role(if index == 0 {
            "sigma-first"
        } else {
            "sigma-last"
        }))
    }
    fn length(&self, object: &Options) -> Result<i64, Error> {
        self.emit(json!(["length", label(object)]))?;
        Ok(4)
    }
    fn tuple(&self, object: &Options) -> Result<Options, Error> {
        Ok(object.clone())
    }
    fn add_one(&self, value: &Options) -> Result<Options, Error> {
        let Options::Atom(V::Int(v)) = value else {
            panic!("own seed")
        };
        Ok(integer(v + 1))
    }
    fn manual_seed(&self, value: &Options) -> Result<Options, Error> {
        self.emit(json!(["seed", encode(value)]))?;
        Ok(role("generator"))
    }
    fn randn_cpu(&self, shape: &Options, generator: &Options) -> Result<Options, Error> {
        self.emit(json!(["randn", encode(shape), encode(generator)]))?;
        Ok(role("random"))
    }
    fn inpaint(&self, model: &Options, sigmas: &Options) -> Result<Options, Error> {
        self.emit(json!(["inpaint", encode(model), encode(sigmas)]))?;
        Ok(role("model-k"))
    }
    fn maximum_denoise(
        &self,
        state: &SamplerState,
        model: &Options,
        sigmas: &Options,
    ) -> Result<Options, Error> {
        self.emit(json!(["maximum", encode(model), encode(sigmas)]))?;
        if self.case["mode"] == "replace-options" {
            state.fields.set(
                "extra_options",
                Options::dictionary(vec![("own".into(), integer(9))]),
            );
        }
        Ok(Options::Atom(V::Bool(false)))
    }
    fn call(
        &self,
        function: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
    ) -> Result<Options, Error> {
        assert!(kwargs.is_empty());
        let name = label(function);
        let mut event = vec![json!(if name == "external" {
            "external"
        } else {
            "call"
        })];
        if name != "external" {
            event.push(json!(name));
        }
        event.extend(args.iter().map(encode));
        self.emit(json!(event))?;
        if name == "external" {
            return Ok(Options::Atom(V::None));
        }
        if name.ends_with(".to") {
            return Ok(role(&format!("{name}:{}", label(&args[0]))));
        }
        Ok(role(if name.ends_with(".noise_scaling") {
            "scaled-noise"
        } else {
            "output"
        }))
    }
    fn call_with_mapping(
        &self,
        function: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
        mapping: &Options,
    ) -> Result<Options, Error> {
        assert_eq!(args.len(), 3);
        assert_eq!(label(&args[0]), "model-k");
        assert_eq!(
            kwargs
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["extra_args", "callback", "disable"]
        );
        let get = |key: &str| {
            kwargs
                .iter()
                .find(|(name, _)| name == key)
                .unwrap()
                .1
                .clone()
        };
        self.emit(json!([
            "sampler",
            label(function),
            encode(&self.model_k),
            encode(&args[1]),
            encode(&args[2]),
            get("extra_args").same_object(&self.extra),
            encode(&get("disable")),
            encode(mapping)
        ]))?;
        let callback = get("callback");
        self.saved.borrow_mut().push(callback.clone());
        if self.case["mode"] == "replace-function" {
            self.state
                .fields
                .set("sampler_function", role("sampler_new"));
        }
        if self.case["mode"] != "deferred" {
            for index in [0, 1] {
                drop(self.invoke(&callback, &self.step("step", index))?);
            }
        }
        if self.case["mode"] == "replace-model" {
            self.fields
                .borrow_mut()
                .insert(("model".into(), "inner_model".into()), role("inner-b"));
        }
        Ok(role("samples"))
    }
    fn callback(&self, function: Callback) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(Function(function))))
    }
    fn detail(&self, args: &[Options]) -> Result<Options, Error> {
        let mut event = vec![json!("detail")];
        event.extend(args.iter().map(encode));
        self.emit(json!(event))?;
        Ok(Options::Atom(V::None))
    }
}
#[test]
fn original_stateful_ksampler_preserves_raw_fields_eager_detail_live_rebinding_random_setup_escaping_callbacks_and_failure_prefixes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-state.json")).unwrap();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let inpaint = Options::dictionary(if case["random"] == true {
            vec![("random".into(), Options::Atom(V::Bool(true)))]
        } else {
            vec![]
        });
        let state = Rc::new(SamplerState::new(
            role("sampler_old"),
            Options::dictionary(vec![("own".into(), integer(1))]),
            inpaint.clone(),
        ));
        let extra = Options::dictionary(if case["mode"] == "missing-seed" {
            vec![]
        } else {
            vec![("seed".into(), integer(901))]
        });
        let shape = Options::Tuple(Rc::new(vec![integer(2), integer(1)]));
        let mut fields = BTreeMap::new();
        for (owner, name, value) in [
            ("model", "inner_model", role("inner-a")),
            ("model", "model_patcher", role("patcher")),
            ("model", "cfg", Options::Atom(V::Float(2.5))),
            ("patcher", "model", role("physical")),
            ("inner-a", "model_sampling", role("sampling-a")),
            ("inner-b", "model_sampling", role("sampling-b")),
            ("noise", "shape", shape.clone()),
            ("noise", "dtype", role("dtype")),
            ("noise", "device", role("device")),
            ("x", "shape", shape.clone()),
            ("denoised", "shape", shape),
        ] {
            fields.insert((owner.into(), name.into()), value);
        }
        let data = Rc::new(Data {
            case: case.clone(),
            state: state.clone(),
            extra: extra.clone(),
            dicts: RefCell::new(vec![
                (extra.clone(), "extra".into()),
                (inpaint, "inpaint".into()),
            ]),
            fields: RefCell::new(fields),
            model_k: Options::dictionary(vec![]),
            saved: RefCell::new(vec![]),
            events: RefCell::new(vec![]),
        });
        let mut output = Value::Null;
        let result = (|| -> Result<(), Error> {
            let value = sampler_state::sample(
                state.clone(),
                sampler_state::Arguments {
                    model: role("model"),
                    sigmas: role("sigmas"),
                    extra: extra.clone(),
                    callback: if case["external"] == true {
                        role("external")
                    } else {
                        Options::Atom(V::None)
                    },
                    noise: role("noise"),
                    latent: role("latent"),
                    mask: role("mask"),
                    disable: Options::Atom(V::Bool(false)),
                },
                data.clone(),
            )?;
            output = encode(&value);
            if matches!(case["mode"].as_str(), Some("retained" | "deferred")) {
                data.emit(json!(["after"]))?;
                let callback = data.saved.borrow()[0].clone();
                drop(data.invoke(&callback, &data.step("after-step", 2))?);
            }
            Ok(())
        })();
        let error = match result {
            Ok(()) => Value::Null,
            Err(Error::ReferenceException { class, message }) => json!([class, message]),
            Err(error) => panic!("unbound stateful sampler {index}: {error:?}"),
        };
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(output, case["output"], "output {index}");
        assert_eq!(
            json!(*data.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(
            json!(extra.get("denoise_mask").unwrap().is_some()),
            case["mask"]
        );
        assert_eq!(
            encode(&state.fields.get("extra_options").unwrap()),
            case["extra_options"]
        );
        assert_eq!(
            encode(&state.fields.get("sampler_function").unwrap()),
            case["sampler_name"]
        );
        // Own fixture teardown, not a claim that generic source cycles are GC'd.
        data.saved.borrow_mut().clear();
    }
}
