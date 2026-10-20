use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::{GeneralMapping, Options, SourceText},
    sampling::guider::{DeviceContext, GuiderState, OuterBindings, SampleArguments, outer_sample},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Copy)]
enum Kind {
    Token,
    Tensor,
    Patcher,
    Pool,
}
struct Object {
    label: String,
    kind: Kind,
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own object")
    };
    value.downcast_ref::<Object>().unwrap()
}
fn labelled(label: &str, kind: Kind) -> Options {
    Options::Object(Rc::new(Object {
        label: label.into(),
        kind,
    }))
}
fn token(label: &str) -> Options {
    labelled(label, Kind::Token)
}
fn tensor(label: &str) -> Options {
    labelled(label, Kind::Tensor)
}
fn patcher(label: &str) -> Options {
    labelled(label, Kind::Patcher)
}
fn pool(label: &str) -> Options {
    labelled(label, Kind::Pool)
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
fn none() -> Options {
    Options::Atom(V::None)
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn tuple(values: Vec<Options>) -> Options {
    Options::Tuple(Rc::new(values))
}
fn failure(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
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
        Options::Atom(V::Text(value)) => json!(value),
        Options::Atom(V::Bool(value)) => json!(value),
        Options::Atom(V::Int(value)) => json!(value),
        Options::List(values) => {
            json!({"list":values.borrow().iter().map(snap).collect::<Vec<_>>()})
        }
        Options::Tuple(values) => json!({"tuple":values.iter().map(snap).collect::<Vec<_>>()}),
        Options::Object(_) => json!({"object":object(value).label}),
        _ => panic!("own snapshot"),
    }
}
struct Bindings<'a> {
    state: GuiderState,
    keys: Rc<OwnMappingKeys>,
    extra: Options,
    action: &'a str,
    failure: Option<usize>,
    exit_failure: bool,
    events: RefCell<Vec<Value>>,
}
impl Bindings<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(failure("ValueError", "own outer-sample failure"));
        }
        Ok(())
    }
    fn mapping(&self, key: &str, value: Options) -> Options {
        let mapping = self.keys.empty();
        mapping.set(key, value).unwrap();
        mapping
    }
}
struct Context<'a, 'b>(&'a Bindings<'b>);
impl DeviceContext for Context<'_, '_> {
    fn enter(&self) -> Result<(), Error> {
        self.0.emit(json!(["enter"]))
    }
    fn exit(&self) -> Result<(), Error> {
        self.0.emit(json!(["exit"]))?;
        if self.0.exit_failure {
            return Err(failure("RuntimeError", "own context exit failure"));
        }
        Ok(())
    }
}
impl OuterBindings for Bindings<'_> {
    fn shape(&self, value: &Options) -> Result<Options, Error> {
        assert!(matches!(object(value).kind, Kind::Tensor));
        self.emit(json!(["shape", object(value).label]))?;
        Ok(tuple(vec![
            Options::Atom(V::Int(2)),
            Options::Atom(V::Int(3)),
        ]))
    }
    fn prepare(
        &self,
        model: &Options,
        shape: &Options,
        conds: &Options,
        options: &Options,
    ) -> Result<Options, Error> {
        self.emit(json!([
            "prepare",
            snap(model),
            snap(shape),
            snap(conds),
            snap(options)
        ]))?;
        Ok(tuple(vec![
            token("inner-model"),
            self.mapping("prepared-cond", token("prepared-condition")),
            list(vec![token("loaded")]),
        ]))
    }
    fn unpack_prepared(&self, value: &Options) -> Result<[Options; 3], Error> {
        let Options::Tuple(values) = value else {
            panic!("own prepared")
        };
        Ok([values[0].clone(), values[1].clone(), values[2].clone()])
    }
    fn load_device(&self, patcher: &Options) -> Result<Options, Error> {
        assert!(matches!(object(patcher).kind, Kind::Patcher));
        self.emit(json!(["device", object(patcher).label]))?;
        Ok(Options::Atom(V::Text(format!(
            "dev:{}",
            object(patcher).label
        ))))
    }
    fn model_dtype(&self, patcher: &Options) -> Result<Options, Error> {
        self.emit(json!(["dtype", object(patcher).label]))?;
        Ok(Options::Atom(V::Text("own-dtype".into())))
    }
    fn multigpu_patchers(
        &self,
        model: &Options,
        loaded: &Options,
        options: &Options,
    ) -> Result<Options, Error> {
        self.emit(json!([
            "multigpu",
            snap(model),
            snap(loaded),
            snap(options)
        ]))?;
        Ok(self.extra.clone())
    }
    fn truth(&self, value: &Options) -> Result<bool, Error> {
        let Options::List(values) = value else {
            panic!("own extra list")
        };
        Ok(!values.borrow().is_empty())
    }
    fn thread_pool(&self, devices: &Options) -> Result<Options, Error> {
        let Options::List(values) = devices else {
            panic!("own devices")
        };
        self.emit(json!([
            "pool",
            values.borrow().iter().map(text).collect::<Vec<_>>()
        ]))?;
        Ok(pool("created-pool"))
    }
    fn device_context(&self, device: &Options) -> Result<Box<dyn DeviceContext + '_>, Error> {
        self.emit(json!(["context", text(device)]))?;
        Ok(Box::new(Context(self)))
    }
    fn move_float32(&self, value: &Options, device: &Options) -> Result<Options, Error> {
        if matches!(value, Options::Atom(V::None)) {
            return Err(failure(
                "AttributeError",
                "'NoneType' object has no attribute 'to'",
            ));
        }
        self.emit(
            json!(["move",object(value).label,[],{"device":text(device),"dtype":"own-f32"}]),
        )?;
        Ok(tensor(&format!("{}:moved", object(value).label)))
    }
    fn move_sigmas(&self, value: &Options, device: &Options) -> Result<Options, Error> {
        self.emit(json!(["move", object(value).label, [text(device)], {}]))?;
        Ok(tensor(&format!("{}:moved", object(value).label)))
    }
    fn cast_options(
        &self,
        options: &Options,
        device: &Options,
        dtype: &Options,
    ) -> Result<(), Error> {
        self.emit(json!(["cast", snap(options), text(device), text(dtype)]))
    }
    fn pre_run(&self, patcher: &Options) -> Result<(), Error> {
        self.emit(json!(["pre-run", object(patcher).label]))?;
        if self.action == "replace-options" && object(patcher).label == "main" {
            let options = self.mapping("replacement", Options::Atom(V::Bool(true)));
            options.set("multigpu_thread_pool", pool("replacement-pool"))?;
            self.state.fields.set("model_options", options);
        }
        if self.action == "clear-extras" && object(patcher).label == "extra:0" {
            let Options::List(values) = &self.extra else {
                unreachable!()
            };
            values.borrow_mut().clear();
        }
        Ok(())
    }
    fn inner_sample(&self, args: &SampleArguments, device: &Options) -> Result<Options, Error> {
        let positional = [
            &args.noise,
            &args.latent_image,
            device,
            &args.sampler,
            &args.sigmas,
            &args.denoise_mask,
            &args.callback,
            &args.disable_pbar,
            &args.seed,
        ];
        self.emit(json!(["inner",positional.iter().map(|v|snap(v)).collect::<Vec<_>>(),{"latent_shapes":snap(&args.latent_shapes)}]))?;
        if self.action == "append-extra" {
            let Options::List(values) = &self.extra else {
                unreachable!()
            };
            values.borrow_mut().push(patcher("extra:2"));
        }
        Ok(token("samples"))
    }
    fn shutdown(&self, pool: &Options) -> Result<(), Error> {
        assert!(matches!(object(pool).kind, Kind::Pool));
        self.emit(json!(["shutdown", object(pool).label]))
    }
    fn cleanup(&self, patcher: &Options) -> Result<(), Error> {
        self.emit(json!(["cleanup", object(patcher).label]))
    }
    fn cleanup_models(&self, conds: &Options, loaded: &Options) -> Result<(), Error> {
        self.emit(json!(["cleanup-models", snap(conds), snap(loaded)]))
    }
}
#[test]
fn original_outer_guider_preserves_prepare_mutable_multigpu_moves_finally_context_cleanup_fields_and_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guider-outer.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 224);
    for (index, case) in cases.iter().enumerate() {
        let bindings = Bindings {
            state: GuiderState::default(),
            keys: OwnMappingKeys::seeded(),
            extra: list(
                (0..case["count"].as_u64().unwrap())
                    .map(|i| patcher(&format!("extra:{i}")))
                    .collect(),
            ),
            action: case["action"].as_str().unwrap(),
            failure: case["failure"].as_u64().map(|v| v as usize),
            exit_failure: case["exit_failure"].as_bool().unwrap(),
            events: RefCell::new(vec![]),
        };
        let options = bindings.mapping("initial", Options::Atom(V::Bool(true)));
        if case["existing_pool"] == true {
            options
                .set("multigpu_thread_pool", pool("existing-pool"))
                .unwrap();
        }
        bindings.state.fields.set("model_patcher", patcher("main"));
        bindings.state.fields.set("model_options", options.clone());
        bindings.state.fields.set(
            "conds",
            bindings.mapping("initial-cond", token("condition")),
        );
        let arguments = SampleArguments {
            noise: tensor("noise"),
            latent_image: if case["latent_none"] == true {
                none()
            } else {
                tensor("latent")
            },
            sampler: token("sampler"),
            sigmas: tensor("sigmas"),
            denoise_mask: token("mask"),
            callback: token("callback"),
            disable_pbar: Options::Atom(V::Bool(true)),
            seed: Options::Atom(V::Int(17)),
            latent_shapes: list(vec![tuple(vec![
                Options::Atom(V::Int(2)),
                Options::Atom(V::Int(3)),
            ])]),
        };
        let result = outer_sample(&bindings.state, arguments, &bindings);
        let (result, error) = match result {
            Ok(value) => (snap(&value), Value::Null),
            Err(error) => match error.root() {
                Error::ReferenceException { class, message } => {
                    (Value::Null, json!([class, message]))
                }
                error => panic!("native outer guider {index}: {error:?}"),
            },
        };
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(result, case["result"], "result {index}");
        let fields = json!({"dict":bindings.state.fields.entries().iter().map(|(k,v)|json!([k,snap(v)])).collect::<Vec<_>>()});
        assert_eq!(fields, case["fields"], "fields {index}");
        assert_eq!(
            snap(&options),
            case["original_options"],
            "original options {index}"
        );
        assert_eq!(snap(&bindings.extra), case["extras"], "extras {index}");
    }
}
