use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        extensions::{Original, Wrapper},
        options::{GeneralMapping, Options, SourceText, copy_nested_dicts},
    },
    sampling::{
        guider::{GuiderState, SampleArguments},
        guider_sample::{self, SampleBindings},
    },
};
use serde_json::{Value, json};
use std::{any::Any, cell::RefCell, rc::Rc};
enum Object {
    Token(String),
    Tensor {
        label: String,
        parts: Option<Options>,
        shape: Options,
    },
    Patcher(RefCell<Options>),
    Record(String, Options),
    Callback(Option<(Options, Options)>),
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own object")
    };
    value.downcast_ref::<Object>().unwrap()
}
fn token(value: &str) -> Options {
    Options::Object(Rc::new(Object::Token(value.into())))
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
fn shape() -> Options {
    tuple(
        [1, 2, 3]
            .iter()
            .map(|v| Options::Atom(V::Int(*v)))
            .collect(),
    )
}
fn tensor(label: &str, parts: Option<Options>, shape: Options) -> Options {
    Options::Object(Rc::new(Object::Tensor {
        label: label.into(),
        parts,
        shape,
    }))
}
fn plain(label: &str) -> Options {
    tensor(label, None, shape())
}
fn label(value: &Options) -> &str {
    match object(value) {
        Object::Token(v) | Object::Tensor { label: v, .. } => v,
        Object::Patcher(_) => "patcher",
        _ => panic!("own label"),
    }
}
fn text(value: &Options) -> String {
    SourceText::get(value).unwrap().utf8().unwrap()
}
fn values(value: &Options) -> Vec<Options> {
    match value {
        Options::List(v) => v.borrow().clone(),
        Options::Tuple(v) => v.as_ref().clone(),
        _ => panic!("own sequence"),
    }
}
fn snap(value: &Options) -> Value {
    if SourceText::get(value).is_some() {
        return json!(text(value));
    }
    if let Some(mapping) = GeneralMapping::get(value) {
        return json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([snap(k),snap(v)])).collect::<Vec<_>>()});
    }
    match value {
        Options::Atom(V::None) => Value::Null,
        Options::Atom(V::Text(v)) => json!(v),
        Options::Atom(V::Int(v)) => json!(v),
        Options::Atom(V::Bool(v)) => json!(v),
        Options::List(v) => json!({"list":v.borrow().iter().map(snap).collect::<Vec<_>>()}),
        Options::Tuple(v) => json!({"tuple":v.iter().map(snap).collect::<Vec<_>>()}),
        Options::Object(value) => {
            if value.downcast_ref::<Wrapper<Options>>().is_some() {
                return json!({"function":"wrapper"});
            }
            match value.downcast_ref::<Object>().unwrap() {
                Object::Record(_, data) => snap(data),
                Object::Callback(packed) => {
                    json!({"function":if packed.is_some() {"callback"} else {"original_callback"}})
                }
                _ => json!({"object":label(&Options::Object(value.clone()))}),
            }
        }
        _ => panic!("own snapshot"),
    }
}
struct Context {
    state: Rc<GuiderState>,
    keys: Rc<OwnMappingKeys>,
    events: Rc<RefCell<Vec<Value>>>,
    failure: Option<usize>,
    action: String,
    groups: usize,
    finally_fail: bool,
}
impl Context {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own sample failure".into(),
            });
        }
        Ok(())
    }
    fn mapping(&self, key: &str, value: Options) -> Options {
        let result = self.keys.empty();
        result.set(key, value).unwrap();
        result
    }
    fn record(&self, name: &str) -> Options {
        let data = self.mapping("label", SourceText::from_utf8(name));
        data.set("shared", token("shared")).unwrap();
        Options::Object(Rc::new(Object::Record(name.into(), data)))
    }
    fn mode(&self) -> Options {
        let patcher = self.state.fields.get("model_patcher").unwrap();
        let Object::Patcher(mode) = object(&patcher) else {
            panic!("own patcher")
        };
        mode.borrow().clone()
    }
    fn nested(&self, value: &Options, shapes: &Options) -> Result<Options, Error> {
        self.emit(json!(["unpack", snap(value), snap(shapes)]))?;
        let parts = list(
            values(shapes)
                .iter()
                .enumerate()
                .map(|(i, shape)| {
                    tensor(&format!("part:{}:{i}", label(value)), None, shape.clone())
                })
                .collect(),
        );
        self.emit(json!(["nested-build", snap(&parts)]))?;
        let name = format!(
            "nested:{}",
            values(&parts)
                .iter()
                .map(label)
                .collect::<Vec<_>>()
                .join(",")
        );
        Ok(tensor(&name, Some(parts), shape()))
    }
    fn callback(&self, callback: &Options, x0: &Options, x: &Options) -> Result<Options, Error> {
        let Object::Callback(packed) = object(callback) else {
            panic!("own callback")
        };
        if let Some((original, shapes)) = packed {
            let x0 = self.nested(x0, shapes)?;
            let x = self.nested(x, shapes)?;
            return self.callback(original, &x0, &x);
        }
        self.emit(json!(["callback", 2, snap(x0), snap(x), 5]))?;
        Ok(token("callback-return"))
    }
}
struct Bindings(Rc<Context>);
impl SampleBindings for Bindings {
    fn empty_sigmas(&self, sigmas: &Options) -> Result<bool, Error> {
        let shape = self.shape(sigmas)?;
        let entries = values(&shape);
        Ok(matches!(entries.last(), Some(Options::Atom(V::Int(0)))))
    }
    fn is_nested(&self, value: &Options) -> Result<bool, Error> {
        self.0.emit(json!(["nested", label(value)]))?;
        let Object::Tensor { parts, .. } = object(value) else {
            panic!("own tensor")
        };
        Ok(parts.is_some())
    }
    fn unbind(&self, value: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["unbind", label(value)]))?;
        let Object::Tensor { parts, .. } = object(value) else {
            panic!("own tensor")
        };
        Ok(list(values(parts.as_ref().unwrap())))
    }
    fn shape(&self, value: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["shape", label(value)]))?;
        let Object::Tensor { shape, .. } = object(value) else {
            panic!("own tensor")
        };
        Ok(shape.clone())
    }
    fn shape_tuple(&self, value: &Options) -> Result<Options, Error> {
        Ok(value.clone())
    }
    fn pack(&self, value: &Options) -> Result<[Options; 2], Error> {
        self.0.emit(json!(["pack", snap(value)]))?;
        let entries = values(value);
        let name = format!(
            "packed:{}",
            entries.iter().map(label).collect::<Vec<_>>().join(",")
        );
        let shapes = list(
            entries
                .iter()
                .map(|v| {
                    let Object::Tensor { shape, .. } = object(v) else {
                        panic!("own tensor")
                    };
                    shape.clone()
                })
                .collect(),
        );
        Ok([plain(&name), shapes])
    }
    fn length(&self, value: &Options) -> Result<usize, Error> {
        Ok(values(value).len())
    }
    fn index(&self, value: &Options, index: usize) -> Result<Options, Error> {
        Ok(values(value)[index].clone())
    }
    fn prefix(&self, value: &Options, length: usize) -> Result<Options, Error> {
        let entries = values(value);
        Ok(list(entries[..length.min(entries.len())].to_vec()))
    }
    fn append(&self, value: &Options, item: Options) -> Result<(), Error> {
        let Options::List(values) = value else {
            panic!("own list")
        };
        values.borrow_mut().push(item);
        Ok(())
    }
    fn set_index(&self, value: &Options, index: usize, item: Options) -> Result<(), Error> {
        let Options::List(values) = value else {
            panic!("own list")
        };
        values.borrow_mut()[index] = item;
        Ok(())
    }
    fn ones(&self, shape: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["ones", snap(shape)]))?;
        Ok(tensor("ones", None, shape.clone()))
    }
    fn load_device(&self, _: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["load-device"]))?;
        Ok(SourceText::from_utf8("load"))
    }
    fn prepare_mask(
        &self,
        mask: &Options,
        shape: &Options,
        device: &Options,
    ) -> Result<Options, Error> {
        self.0.emit(json!([
            "prepare-mask",
            snap(mask),
            snap(shape),
            text(device)
        ]))?;
        Ok(tensor(
            &format!("prepared:{}", label(mask)),
            None,
            shape.clone(),
        ))
    }
    fn mask_float(&self, mask: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["mask-float", label(mask)]))?;
        Ok(plain(&format!("float:{}", label(mask))))
    }
    fn detail(&self, _: &Options, shapes: &Options) -> Result<(), Error> {
        self.0.emit(json!(["detail", "Model", snap(shapes)]))
    }
    fn packed_callback(&self, callback: &Options, shapes: &Options) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(Object::Callback(Some((
            callback.clone(),
            shapes.clone(),
        ))))))
    }
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self.0.keys.empty())
    }
    fn copy_condition_record(&self, record: &Options) -> Result<Options, Error> {
        let Object::Record(label, data) = object(record) else {
            panic!("own record")
        };
        self.0.emit(json!(["record-copy", label]))?;
        if self.0.action == "replace-cond-root" && label == "positive:0" {
            self.0
                .state
                .fields
                .set("conds", self.0.mapping("replacement", list(vec![])));
        }
        if self.0.action == "append-condition" && label == "positive:0" {
            self.0
                .state
                .fields
                .get("original_conds")?
                .set("extra", list(vec![self.0.record("extra:0")]))?;
        }
        data.shallow_copy_dict()
    }
    fn preprocess_hooks(&self, conditions: &Options) -> Result<(), Error> {
        self.0.emit(json!(["preprocess-hooks", snap(conditions)]))
    }
    fn clone_model_options(&self, options: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["clone-options", snap(options)]))?;
        copy_nested_dicts(options)
    }
    fn hook_mode(&self, _: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["hook-get"]))?;
        Ok(self.0.mode())
    }
    fn total_hook_groups(&self, conditions: &Options) -> Result<usize, Error> {
        self.0.emit(json!(["hook-count", snap(conditions)]))?;
        Ok(self.0.groups)
    }
    fn min_vram_mode(&self) -> Result<Options, Error> {
        Ok(token("min-vram"))
    }
    fn set_hook_mode(&self, patcher: &Options, mode: &Options) -> Result<(), Error> {
        self.0.emit(json!(["hook-set", snap(mode)]))?;
        let Object::Patcher(target) = object(patcher) else {
            panic!("own patcher")
        };
        *target.borrow_mut() = mode.clone();
        Ok(())
    }
    fn prepare_model(
        &self,
        _: &Options,
        conditions: &Options,
        options: &Options,
    ) -> Result<(), Error> {
        self.0
            .emit(json!(["prepare-model", snap(conditions), snap(options)]))
    }
    fn filter_hooks(&self, conditions: &Options, options: &Options) -> Result<(), Error> {
        self.0
            .emit(json!(["filter-hooks", snap(conditions), snap(options)]))
    }
    fn outer_method(&self) -> Result<Original<Options>, Error> {
        let context = self.0.clone();
        Ok(Rc::new(move |args, kwargs| {
            let keyword_values = kwargs
                .iter()
                .map(|(k, v)| (k.clone(), snap(v)))
                .collect::<serde_json::Map<_, _>>();
            context.emit(json!([
                "outer",
                args.iter().map(snap).collect::<Vec<_>>(),
                keyword_values
            ]))?;
            if !matches!(args[5], Options::Atom(V::None)) {
                let result = context.callback(&args[5], &plain("x0"), &plain("x"))?;
                context.emit(json!(["callback-result", snap(&result)]))?;
            }
            Ok(plain("output"))
        }))
    }
    fn canonical_guider(&self) -> Rc<dyn Any> {
        self.0.state.clone()
    }
    fn offload_device(&self, _: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["offload-device"]))?;
        Ok(SourceText::from_utf8("offload"))
    }
    fn cast_offload_options(&self, options: &Options, device: &Options) -> Result<(), Error> {
        self.0
            .emit(json!(["cast-offload", snap(options), text(device)]))?;
        if self.0.finally_fail {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: "own finally cast failure".into(),
            });
        }
        Ok(())
    }
    fn restore_hook_patches(&self, _: &Options) -> Result<(), Error> {
        self.0.emit(json!(["restore-hooks"]))
    }
    fn unpack_nested(&self, output: &Options, shapes: &Options) -> Result<Options, Error> {
        self.0.nested(output, shapes)
    }
}
fn latent(name: &str, kind: &str) -> Options {
    if kind == "flat" {
        return plain(name);
    }
    let count = match kind {
        "nested-zero" => 0,
        "nested-one" => 1,
        "nested-two" => 2,
        "nested-three" => 3,
        _ => panic!("own nested kind"),
    };
    tensor(
        name,
        Some(list(
            (0..count).map(|i| plain(&format!("{name}:{i}"))).collect(),
        )),
        shape(),
    )
}
#[test]
fn original_public_sample_preserves_nested_geometry_masks_callbacks_condition_copies_hook_restore_partial_fields_and_finally_error_precedence()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guider-sample.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 195);
    for (index, case) in cases.iter().enumerate() {
        let context = Rc::new(Context {
            state: Rc::new(GuiderState::default()),
            keys: OwnMappingKeys::seeded(),
            events: Rc::new(RefCell::new(vec![])),
            failure: case["failure"].as_u64().map(|v| v as usize),
            action: case["action"].as_str().unwrap().into(),
            groups: case["groups"].as_u64().unwrap() as usize,
            finally_fail: case["finally_fail"].as_bool().unwrap(),
        });
        let bindings = Bindings(context.clone());
        context.state.fields.set(
            "model_patcher",
            Options::Object(Rc::new(Object::Patcher(RefCell::new(token(
                "original-mode",
            ))))),
        );
        let conditions = context.mapping(
            "positive",
            list(vec![
                context.record("positive:0"),
                context.record("positive:1"),
            ]),
        );
        conditions
            .set("negative", list(vec![context.record("negative:0")]))
            .unwrap();
        context.state.fields.set("original_conds", conditions);
        let registry = context.keys.empty();
        let mut wrappers = vec![];
        if case["wrapped"] == true {
            let events = context.events.clone();
            let failure = context.failure;
            let state = Rc::downgrade(&context.state);
            let wrapper: Wrapper<Options> = Rc::new(move |executor, args, kwargs| {
                let actual = executor
                    .class_obj
                    .as_ref()
                    .unwrap()
                    .clone()
                    .downcast::<GuiderState>()
                    .unwrap();
                assert!(Rc::ptr_eq(&actual, &state.upgrade().unwrap()));
                let mut events_ref = events.borrow_mut();
                events_ref.push(json!(["wrapper"]));
                let fail = failure == Some(events_ref.len() - 1);
                drop(events_ref);
                if fail {
                    return Err(Error::ReferenceException {
                        class: "ValueError",
                        message: "own sample failure".into(),
                    });
                }
                executor.next(args, kwargs)
            });
            wrappers.push(Options::Object(Rc::new(wrapper)));
        }
        registry.set_item(none(), list(wrappers)).unwrap();
        let options = context.mapping("marker", token("original-options"));
        options
            .set("shared-list", list(vec![token("shared")]))
            .unwrap();
        options
            .set(
                "transformer_options",
                context.mapping("wrappers", context.mapping("outer_sample", registry)),
            )
            .unwrap();
        context.state.fields.set("model_options", options.clone());
        let latent_kind = case["latent_kind"].as_str().unwrap();
        let original_latent = latent("latent", latent_kind);
        let mask_kind = case["mask_kind"].as_str().unwrap();
        let arguments = SampleArguments {
            noise: latent("noise", latent_kind),
            latent_image: original_latent.clone(),
            sampler: token("sampler"),
            sigmas: tensor(
                "sigmas",
                None,
                tuple(vec![Options::Atom(V::Int(if case["empty"] == true {
                    0
                } else {
                    2
                }))]),
            ),
            denoise_mask: if mask_kind == "none" {
                none()
            } else {
                latent("mask", mask_kind)
            },
            callback: if case["callback_present"] == true {
                Options::Object(Rc::new(Object::Callback(None)))
            } else {
                none()
            },
            disable_pbar: Options::Atom(V::Bool(true)),
            seed: Options::Atom(V::Int(17)),
            latent_shapes: none(),
        };
        let result = guider_sample::sample(&context.state, arguments, &bindings);
        let (result, error, identity) = match result {
            Ok(value) => (
                snap(&value),
                Value::Null,
                value.same_object(&original_latent),
            ),
            Err(error) => match error.root() {
                Error::ReferenceException { class, message } => {
                    (Value::Null, json!([class, message]), false)
                }
                error => panic!("native sample {index}: {error:?}"),
            },
        };
        assert_eq!(
            json!(*context.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(result, case["result"], "result {index}");
        let fields = json!({"dict":context.state.fields.entries().iter().map(|(k,v)|json!([k,snap(v)])).collect::<Vec<_>>()});
        assert_eq!(fields, case["fields"], "fields {index}");
        assert_eq!(snap(&context.mode()), case["mode"], "mode {index}");
        assert_eq!(
            json!(
                context
                    .state
                    .fields
                    .get("model_options")
                    .unwrap()
                    .same_object(&options)
            ),
            case["options_is_original"],
            "options alias {index}"
        );
        assert_eq!(
            snap(&options),
            case["original_options"],
            "original options {index}"
        );
        assert_eq!(
            json!(identity),
            case["result_is_original_latent"],
            "early return identity {index}"
        );
    }
}
