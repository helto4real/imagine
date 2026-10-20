use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::{GeneralMapping, Options, SourceText, copy_nested_dicts},
    sampling::guider::{
        GuiderState, InnerBindings, ProcessConditions, SampleArguments, inner_sample,
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};

enum Object {
    Token(String),
    Tensor(String),
    Model(String, RefCell<Options>),
    Out(Options),
    Executor,
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own object")
    };
    value.downcast_ref::<Object>().unwrap()
}
fn label(value: &Options) -> &str {
    match object(value) {
        Object::Token(v) | Object::Tensor(v) | Object::Model(v, _) => v,
        _ => panic!("own label"),
    }
}
fn token(value: &str) -> Options {
    Options::Object(Rc::new(Object::Token(value.into())))
}
fn tensor(value: &str) -> Options {
    Options::Object(Rc::new(Object::Tensor(value.into())))
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn text(value: &Options) -> String {
    if let Some(value) = SourceText::get(value) {
        return value.utf8().unwrap();
    }
    let Options::Atom(V::Text(value)) = value else {
        panic!("own text")
    };
    value.clone()
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn snap(value: &Options) -> Value {
    if SourceText::get(value).is_some() {
        return json!(text(value));
    }
    if let Some(mapping) = GeneralMapping::get(value) {
        return json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([text(k),snap(v)])).collect::<Vec<_>>()});
    }
    match value {
        Options::Atom(V::None) => Value::Null,
        Options::Atom(V::Bool(v)) => json!(v),
        Options::Atom(V::Int(v)) => json!(v),
        Options::Atom(V::Text(v)) => json!(v),
        Options::List(values) => {
            json!({"list":values.borrow().iter().map(snap).collect::<Vec<_>>()})
        }
        Options::Tuple(values) => json!({"tuple":values.iter().map(snap).collect::<Vec<_>>()}),
        Options::Object(_) => json!({"object":label(value)}),
        _ => panic!("own snapshot"),
    }
}
struct Bindings<'a> {
    state: GuiderState,
    keys: Rc<OwnMappingKeys>,
    latent_kind: &'a str,
    wrapped: bool,
    action: &'a str,
    failure: Option<usize>,
    events: RefCell<Vec<Value>>,
    first: Options,
    second: Options,
    observed: RefCell<Option<Options>>,
}
impl Bindings<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own inner-sample failure".into(),
            });
        }
        Ok(())
    }
    fn mapping(&self, key: &str, value: Options) -> Options {
        let result = self.keys.empty();
        result.set(key, value).unwrap();
        result
    }
    fn replace_model(&self, action: &str) {
        if self.action == action {
            self.state.fields.set("inner_model", self.second.clone());
        }
    }
}
impl InnerBindings for Bindings<'_> {
    fn set_latent_shapes(&self, model: &Options, shapes: &Options) -> Result<(), Error> {
        self.emit(json!(["set-shapes", label(model), snap(shapes)]))?;
        let Object::Model(_, target) = object(model) else {
            panic!("own model")
        };
        *target.borrow_mut() = shapes.clone();
        Ok(())
    }
    fn count_nonzero_positive(&self, latent: &Options) -> Result<bool, Error> {
        self.emit(json!(["count-nonzero", snap(latent)]))?;
        Ok(self.latent_kind == "positive")
    }
    fn process_latent_in(&self, model: &Options, latent: &Options) -> Result<Options, Error> {
        self.emit(json!(["latent-in", label(model), snap(latent)]))?;
        self.replace_model("in-replace-model");
        Ok(tensor("latent:processed"))
    }
    fn process_conditions(&self, args: ProcessConditions<'_>) -> Result<Options, Error> {
        let values = [
            args.model,
            args.noise,
            args.conditions,
            args.device,
            args.latent,
            args.mask,
            args.seed,
        ];
        self.emit(json!(["process-conditions",values.iter().map(|v|snap(v)).collect::<Vec<_>>(),{"latent_shapes":snap(args.latent_shapes)}]))?;
        self.replace_model("process-replace-model");
        Ok(self.mapping("prepared-cond", token("prepared-condition")))
    }
    fn clone_model_options(&self, options: &Options) -> Result<Options, Error> {
        self.emit(json!(["clone-options", snap(options)]))?;
        if self.action == "copy-replace-options" {
            self.state.fields.set(
                "model_options",
                self.mapping("replacement", Options::Atom(V::Bool(true))),
            );
        }
        copy_nested_dicts(options)
    }
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self.keys.empty())
    }
    fn setdefault(&self, options: &Options, key: &str, default: Options) -> Result<Options, Error> {
        if let Some(value) = options.get(key)? {
            return Ok(value);
        }
        options.set(key, default.clone())?;
        Ok(default)
    }
    fn extra_arguments(&self, options: Options, seed: Options) -> Result<Options, Error> {
        let result = self.mapping("model_options", options);
        result.set("seed", seed)?;
        Ok(result)
    }
    fn sampler_method(&self, sampler: &Options) -> Result<Options, Error> {
        self.emit(json!(["sample-method", label(sampler)]))?;
        Ok(token("sample-method"))
    }
    fn sampler_wrappers(&self, options: &Options) -> Result<Options, Error> {
        self.emit(json!(["wrappers", "sampler_sample", snap(options), true]))?;
        Ok(list(if self.wrapped {
            vec![token("wrapper:0"), token("wrapper:1")]
        } else {
            vec![]
        }))
    }
    fn executor(
        &self,
        method: &Options,
        sampler: &Options,
        wrappers: &Options,
    ) -> Result<Options, Error> {
        self.emit(json!([
            "executor",
            snap(method),
            snap(sampler),
            snap(wrappers)
        ]))?;
        Ok(Options::Object(Rc::new(Object::Executor)))
    }
    fn execute(
        &self,
        executor: &Options,
        args: &SampleArguments,
        extra: &Options,
    ) -> Result<Options, Error> {
        assert!(matches!(object(executor), Object::Executor));
        let values = [
            &args.sigmas,
            extra,
            &args.callback,
            &args.noise,
            &args.latent_image,
            &args.denoise_mask,
            &args.disable_pbar,
        ];
        let mut event = vec![json!("execute")];
        event.extend(values.iter().map(|v| snap(v)));
        self.emit(json!(event))?;
        *self.observed.borrow_mut() = Some(extra.clone());
        self.replace_model("execute-replace-model");
        Ok(tensor("samples"))
    }
    fn samples_float32(&self, samples: &Options) -> Result<Options, Error> {
        self.emit(json!(["samples-to", label(samples), ["own-f32"], {}]))?;
        self.replace_model("cast-replace-model");
        Ok(tensor("samples:f32"))
    }
    fn latent_out_method(&self, model: &Options) -> Result<Options, Error> {
        self.emit(json!(["out-method", label(model)]))?;
        Ok(Options::Object(Rc::new(Object::Out(model.clone()))))
    }
    fn process_latent_out(&self, method: &Options, samples: &Options) -> Result<Options, Error> {
        let Object::Out(model) = object(method) else {
            panic!("own bound method")
        };
        self.emit(json!(["latent-out", label(model), snap(samples)]))?;
        Ok(token(&format!("decoded:{}", label(model))))
    }
}
#[test]
fn original_inner_guider_preserves_nonzero_latents_condition_rebinding_native_copy_sample_sigma_aliases_wrappers_and_bound_decoder_order()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guider-inner.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 209);
    for (index, case) in cases.iter().enumerate() {
        let bindings = Bindings {
            state: GuiderState::default(),
            keys: OwnMappingKeys::seeded(),
            latent_kind: case["latent_kind"].as_str().unwrap(),
            wrapped: case["wrapped"].as_bool().unwrap(),
            action: case["action"].as_str().unwrap(),
            failure: case["failure"].as_u64().map(|v| v as usize),
            events: RefCell::new(vec![]),
            first: Options::Object(Rc::new(Object::Model(
                "first-model".into(),
                RefCell::new(none()),
            ))),
            second: Options::Object(Rc::new(Object::Model(
                "second-model".into(),
                RefCell::new(none()),
            ))),
            observed: RefCell::new(None),
        };
        let options = if case["option_kind"] == "none" {
            none()
        } else {
            let options = bindings.mapping("shared-list", list(vec![token("shared")]));
            if case["option_kind"] == "existing" {
                options
                    .set(
                        "transformer_options",
                        bindings.mapping("kept", token("kept")),
                    )
                    .unwrap();
            }
            if case["option_kind"] == "invalid-transformer" {
                options.set("transformer_options", none()).unwrap();
            }
            options
        };
        bindings
            .state
            .fields
            .set("inner_model", bindings.first.clone());
        bindings.state.fields.set(
            "conds",
            bindings.mapping("initial-cond", token("condition")),
        );
        bindings.state.fields.set("model_options", options.clone());
        let sigmas = tensor("sigmas");
        let args = SampleArguments {
            noise: tensor("noise"),
            latent_image: if bindings.latent_kind == "none" {
                none()
            } else {
                tensor("latent")
            },
            sampler: token("sampler"),
            sigmas: sigmas.clone(),
            denoise_mask: token("mask"),
            callback: token("callback"),
            disable_pbar: Options::Atom(V::Bool(true)),
            seed: Options::Atom(V::Int(17)),
            latent_shapes: list(vec![Options::Tuple(Rc::new(vec![
                Options::Atom(V::Int(2)),
                Options::Atom(V::Int(3)),
            ]))]),
        };
        let result = inner_sample(
            &bindings.state,
            args,
            &Options::Atom(V::Text("own-device".into())),
            &bindings,
        );
        let (result, error) = match result {
            Ok(result) => (snap(&result), Value::Null),
            Err(Error::ReferenceException { class, message }) => {
                (Value::Null, json!([class, message]))
            }
            Err(error) => panic!("native inner guider {index}: {error:?}"),
        };
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(result, case["result"], "result {index}");
        assert_eq!(
            json!(label(&bindings.state.fields.get("inner_model").unwrap())),
            case["model"],
            "model {index}"
        );
        assert_eq!(
            snap(&bindings.state.fields.get("conds").unwrap()),
            case["conds"],
            "conds {index}"
        );
        assert_eq!(
            snap(&bindings.state.fields.get("model_options").unwrap()),
            case["options"],
            "options {index}"
        );
        let shapes = [&bindings.first, &bindings.second]
            .iter()
            .map(|v| {
                let Object::Model(_, shapes) = object(v) else {
                    unreachable!()
                };
                snap(&shapes.borrow())
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(shapes), case["shapes"], "shapes {index}");
        let observed = bindings.observed.borrow();
        let aliases = if let Some(extra) = observed.as_ref() {
            let cloned = extra.get("model_options").unwrap().unwrap();
            let original_list = options.get("shared-list").unwrap().unwrap();
            let copied_list = cloned.get("shared-list").unwrap().unwrap();
            let (Options::List(original_items), Options::List(copied_items)) =
                (&original_list, &copied_list)
            else {
                unreachable!()
            };
            json!({"options_is_original":cloned.same_object(&options),"list_is_original":copied_list.same_object(&original_list),
                "member_is_original":copied_items.borrow()[0].same_object(&original_items.borrow()[0]),
                "sigma_is_original":cloned.get("transformer_options").unwrap().unwrap().get("sample_sigmas").unwrap().unwrap().same_object(&sigmas)})
        } else {
            Value::Null
        };
        assert_eq!(aliases, case["aliases"], "aliases {index}");
        assert_eq!(
            observed.as_ref().map_or(Value::Null, snap),
            case["observed"],
            "observed {index}"
        );
    }
}
