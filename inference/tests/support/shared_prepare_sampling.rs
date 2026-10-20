use super::{
    binding_tests::{check, reference_error, restore},
    kitchen,
};
use imagine_inference::{
    conditioning_preprocess::Dimensions,
    execution_value::Value as V,
    native::Error,
    patching::{
        hooks::{Hook, HookGroup, Kind},
        options::Options,
    },
    sampling::preparation::{self as p, Dependencies, MemoryEstimator, Preparation},
};
use serde_json::{Value, json};
use std::{borrow::Cow, cell::RefCell, rc::Rc};
use tch::{Device, Kind as Dtype};
fn text(value: &str) -> Options {
    Options::Atom(V::Text(value.into()))
}
fn sequence(value: &Options) -> Vec<Options> {
    match value {
        Options::List(v) => v.borrow().clone(),
        Options::Tuple(v) => v.as_ref().clone(),
        _ => panic!("Own sequence"),
    }
}
fn string(value: &Options) -> String {
    let V::Text(v) = value.snapshot().unwrap() else {
        panic!("Own text")
    };
    v
}
struct Controller<'a> {
    case: Cow<'a, Value>,
    event: usize,
    real: String,
}
impl Controller<'_> {
    fn next(&mut self, tag: &str) -> Value {
        let expected = self.case["events"][self.event].clone();
        self.event += 1;
        assert_eq!(expected[0], tag, "{}", self.case);
        expected
    }
    fn boundary(&self, pattern: &str) -> Result<(), Error> {
        if self.case["pattern"] == format!("{pattern}_error") {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: format!("own {pattern} boundary"),
            });
        }
        Ok(())
    }
}
impl MemoryEstimator for Controller<'_> {
    fn extra_conds_shapes(&mut self, record: &Options) -> Result<Options, Error> {
        let expected = self.next("extra");
        assert_eq!(
            string(&record.get("label")?.unwrap()),
            expected[1].as_str().unwrap()
        );
        Ok(record.get("dims")?.unwrap())
    }
    fn memory_required(&mut self, shape: &[i64], conditions: &Options) -> Result<f64, Error> {
        let expected = self.next("memory");
        assert_eq!(json!(shape), expected[1]);
        check(
            &conditions.snapshot()?,
            &expected[2],
            Device::Cpu,
            "whole prepare shape map",
        );
        self.boundary("estimate")?;
        let Options::Dict(values) = conditions else {
            panic!("Own map")
        };
        let mut extra = 0_i64;
        for (_, shapes) in values.borrow().iter() {
            for shape in sequence(shapes) {
                extra += sequence(&shape)
                    .iter()
                    .map(|v| {
                        let V::Int(v) = v.snapshot().unwrap() else {
                            panic!("Own dimension")
                        };
                        v
                    })
                    .product::<i64>();
            }
        }
        Ok(shape.iter().product::<i64>() as f64 * 4. + extra as f64 * 0.125)
    }
}
impl Dependencies for Controller<'_> {
    fn control_models(&mut self, control: &Options) -> Result<Vec<Options>, Error> {
        if matches!(control, Options::Atom(V::None)) {
            return Err(Error::ReferenceException {
                class: "AttributeError",
                message: "'NoneType' object has no attribute 'get_models'".into(),
            });
        }
        assert_eq!(string(control), "control");
        self.next("control_models");
        Ok(vec![text("control"), text("control")])
    }
    fn inference_memory_requirements(
        &mut self,
        _control: &Options,
        dtype: Dtype,
    ) -> Result<f64, Error> {
        let expected = self.next("control_memory");
        assert_eq!(dtype, Dtype::BFloat16);
        assert_eq!(expected[1], "torch.bfloat16");
        Ok(64.)
    }
    fn cleanup_if_present(&mut self, _model: &Options) -> Result<(), Error> {
        panic!("Preparation must not clean up dependencies")
    }
}
impl Preparation for Controller<'_> {
    fn match_multigpu_clones(&mut self, model: &Options) -> Result<(), Error> {
        assert_eq!(string(model), "model");
        self.next("match");
        Ok(())
    }
    fn preprocess_multigpu_conds(
        &mut self,
        conditions: &Options,
        _model: &Options,
        _options: Option<&Options>,
    ) -> Result<(), Error> {
        self.next("preprocess");
        if self.case["pattern"] == "preprocess_append" {
            let Options::List(records) = conditions.get("positive")?.unwrap() else {
                panic!("Own records")
            };
            records.borrow_mut().push(Options::from_value(V::Dict(vec![
                ("label".into(), V::Text("added".into())),
                ("dims".into(), V::Dict(vec![])),
                (
                    "additional_models".into(),
                    V::List(vec![V::Text("preprocessed".into())]),
                ),
            ])));
        }
        self.boundary("preprocess")?;
        if self.case["pattern"] == "control_error" {
            sequence(&conditions.get("positive")?.unwrap())[0]
                .set("control", Options::Atom(V::None))?;
        }
        Ok(())
    }
    fn model_dtype(&mut self, _model: &Options) -> Result<Dtype, Error> {
        self.next("dtype");
        self.boundary("dtype")?;
        Ok(Dtype::BFloat16)
    }
    fn nested_additional_models(&mut self, _model: &Options) -> Result<Vec<Options>, Error> {
        self.next("nested");
        self.boundary("nested")?;
        Ok(vec![text("nested"), text("nested")])
    }
    fn load_models_gpu(
        &mut self,
        models: &[Options],
        required: f64,
        minimum: Option<f64>,
        full: &Options,
    ) -> Result<(), Error> {
        let expected = self.next("load");
        assert_eq!(
            json!(models.iter().map(string).collect::<Vec<_>>()),
            expected[1]
        );
        assert_eq!(
            json!({"memory_required":required,"minimum_memory_required":minimum,"force_full_load":match full.snapshot().unwrap(){V::Bool(v)=>json!(v),V::None=>Value::Null,V::Int(v)=>json!(v),_=>panic!("Own full flag")}}),
            expected[2]
        );
        if self.case["pattern"] == "replace_real" || self.case["pattern"] == "load_error" {
            self.real = "real-replaced".into();
        }
        self.boundary("load")
    }
    fn real_model(&mut self, _model: &Options) -> Result<Options, Error> {
        Ok(text(&self.real))
    }
}
fn options(mode: &str) -> Option<Options> {
    if mode == "none" {
        return None;
    }
    let value = Options::from_value(V::Dict(vec![]));
    if mode == "null_hooks" {
        value
            .set("registered_hooks", Options::Atom(V::None))
            .unwrap();
    }
    if mode == "hooks" {
        let mut hook = Hook::new(Kind::AdditionalModels);
        hook.values = Options::from_value(V::List(vec![
            V::Text("hook".into()),
            V::Text("hook".into()),
        ]));
        let mut group = HookGroup::default();
        group.add(Rc::new(RefCell::new(hook)));
        value
            .set(
                "registered_hooks",
                Options::Object(Rc::new(RefCell::new(group))),
            )
            .unwrap();
    }
    Some(value)
}
#[test]
fn whole_prepare_sampling_preserves_order_duplicates_memory_arguments_and_partial_failures() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-prepare-sampling.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 160);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let mut controller = Controller {
            case: Cow::Borrowed(case),
            event: 0,
            real: "real-original".into(),
        };
        let conditions = Options::from_value(restore(&case["initial"], Device::Cpu));
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>();
        let options = options(case["option_mode"].as_str().unwrap());
        let result = p::prepare_unwrapped(
            &mut controller,
            &text("model"),
            Dimensions::Tuple(&shape),
            &conditions,
            options.as_ref(),
            case["full"].as_bool().unwrap(),
            case["force_offload"].as_bool().unwrap(),
        );
        let note = format!("whole preparation {index}: {}", case["pattern"]);
        match result {
            Ok(out) => {
                assert!(!case["error"].is_string(), "Expected error {note}");
                assert_eq!(
                    out.conditions.same_object(&conditions),
                    case["conditions_same"].as_bool().unwrap()
                );
                check(
                    &V::Tuple(vec![
                        out.real_model.snapshot().unwrap(),
                        out.conditions.snapshot().unwrap(),
                        V::List(out.models.iter().map(|v| v.snapshot().unwrap()).collect()),
                    ]),
                    &case["output"],
                    Device::Cpu,
                    &note,
                );
            }
            Err(error) => {
                assert!(
                    case["error"].is_string(),
                    "Unexpected error {note}: {error}"
                );
                reference_error(error, case);
            }
        }
        assert_eq!(
            controller.event,
            case["events"].as_array().unwrap().len(),
            "events {note}"
        );
        assert_eq!(controller.real, case["real_after"].as_str().unwrap());
        check(
            &conditions.snapshot().unwrap(),
            &case["post"],
            Device::Cpu,
            &format!("post {note}"),
        );
    }
}

fn dict(values: Vec<(&str, Options)>) -> Options {
    Options::Dict(Rc::new(RefCell::new(
        values
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )))
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn registry(options: &Options) -> Options {
    options
        .get("transformer_options")
        .unwrap()
        .unwrap()
        .get("wrappers")
        .unwrap()
        .unwrap()
        .get("prepare_sampling")
        .unwrap()
        .unwrap()
}
fn wrapped_options(style: &str, events: Rc<RefCell<Vec<Value>>>) -> Options {
    use imagine_inference::patching::extensions::{Arguments, Wrapper, WrapperExecutor};
    let last_events = events.clone();
    let last: Wrapper<Options> = Rc::new(move |executor, args, kwargs| {
        last_events
            .borrow_mut()
            .push(json!(["last", executor.idx, executor.class_obj.is_none()]));
        let out = executor.next(args, kwargs)?;
        last_events.borrow_mut().push(json!(["last_return"]));
        Ok(out)
    });
    let last_for_first = last.clone();
    let style_owned = style.to_owned();
    let first: Wrapper<Options> = Rc::new(
        move |executor: &WrapperExecutor<Options>,
              args: &[Options],
              kwargs: &Arguments<Options>| {
            let style = style_owned.as_str();
            events
                .borrow_mut()
                .push(json!(["first", executor.idx, executor.class_obj.is_none()]));
            if matches!(style, "skip" | "skip_bad_callable") {
                return Ok(dict(vec![("short", text("own"))]));
            }
            if style == "raise" {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "own wrapper boundary".into(),
                });
            }
            let mut args = args.to_vec();
            let mut kwargs = kwargs.clone();
            let set = |kwargs: &mut Arguments<Options>, name: &str, value| {
                if let Some((_, v)) = kwargs.iter_mut().find(|(key, _)| key == name) {
                    *v = value;
                } else {
                    kwargs.push((name.into(), value));
                }
            };
            match style {
                "replace_options" => set(&mut kwargs, "force_full_load", Options::Atom(V::None)),
                "empty_shape_offload" => {
                    args[1] = Options::Atom(V::None);
                    set(&mut kwargs, "force_offload", Options::Atom(V::Bool(true)));
                }
                "arg_few" => args.truncate(1),
                "arg_many" => {
                    args.extend([
                        Options::Atom(V::None),
                        Options::Atom(V::Bool(false)),
                        Options::Atom(V::Bool(false)),
                        text("extra"),
                    ]);
                    kwargs.clear();
                }
                "duplicate" => set(&mut kwargs, "model", args[0].clone()),
                "unexpected" => set(&mut kwargs, "unknown", Options::Atom(V::Bool(true))),
                "append" | "replace_list" => {
                    let options = &kwargs
                        .iter()
                        .find(|(key, _)| key == "model_options")
                        .unwrap()
                        .1;
                    let registry = registry(options);
                    if style == "append" {
                        let Options::List(values) = registry.get("second")?.unwrap() else {
                            panic!("Own wrapper list")
                        };
                        values
                            .borrow_mut()
                            .push(Options::Object(Rc::new(last_for_first.clone())));
                    } else {
                        registry.set("second", list(vec![Options::Atom(V::None)]))?;
                    }
                }
                _ => {}
            }
            let mut out = executor.next(&args, &kwargs)?;
            if style == "repeat" {
                out = executor.next(&args, &kwargs)?;
            }
            events.borrow_mut().push(json!(["first_return"]));
            Ok(out)
        },
    );
    let none = || Options::Atom(V::None);
    match style {
        "plain" => dict(vec![]),
        "none_options" => none(),
        "null_transformer" => dict(vec![("transformer_options", none())]),
        "null_wrappers" => dict(vec![(
            "transformer_options",
            dict(vec![("wrappers", none())]),
        )]),
        _ => {
            let groups = if style == "null_registry" {
                none()
            } else {
                dict(vec![
                    (
                        "first",
                        if style == "bad_group" {
                            none()
                        } else {
                            list(vec![if style == "bad_callable" {
                                none()
                            } else {
                                Options::Object(Rc::new(first))
                            }])
                        },
                    ),
                    (
                        "second",
                        list(
                            if matches!(style, "two" | "repeat" | "append" | "replace_list") {
                                vec![Options::Object(Rc::new(last))]
                            } else if style == "skip_bad_callable" {
                                vec![none()]
                            } else {
                                vec![]
                            },
                        ),
                    ),
                ])
            };
            dict(vec![(
                "transformer_options",
                dict(vec![("wrappers", dict(vec![("prepare_sampling", groups)]))]),
            )])
        }
    }
}
#[test]
fn whole_preparation_wrappers_preserve_rebinding_shared_options_short_circuit_and_argument_errors()
{
    use imagine_inference::sampling::preparation_wrappers as w;
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-prepare-wrappers.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 42);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let controller = Rc::new(RefCell::new(Controller {
            case: Cow::Owned(case.clone()),
            event: 0,
            real: "real-original".into(),
        }));
        let events = Rc::new(RefCell::new(vec![]));
        let options = wrapped_options(case["style"].as_str().unwrap(), events.clone());
        let conditions = dict(vec![("positive", list(vec![])), ("negative", list(vec![]))]);
        let result = w::prepare(
            controller.clone(),
            w::Request {
                model: text("model"),
                noise_shape: Options::from_value(V::Tuple(vec![
                    V::Int(2),
                    V::Int(4),
                    V::Int(3),
                    V::Int(4),
                ])),
                conditions: conditions.clone(),
                model_options: options.clone(),
                force_full_load: false,
                force_offload: false,
            },
        );
        let note = format!(
            "preparation wrapper {index}: {} {}",
            case["pattern"], case["style"]
        );
        match result {
            Ok(out) => {
                assert!(!case["error"].is_string(), "Expected error {note}");
                if let Options::Tuple(values) = &out {
                    assert_eq!(
                        values[1].same_object(&conditions),
                        case["conditions_same"].as_bool().unwrap()
                    );
                }
                check(
                    &out.snapshot().unwrap(),
                    &case["output"],
                    Device::Cpu,
                    &note,
                );
            }
            Err(error) => {
                assert!(
                    case["error"].is_string(),
                    "Unexpected error {note}: {error}"
                );
                reference_error(error, case);
            }
        }
        assert_eq!(
            json!(*events.borrow()),
            case["wrapper_events"],
            "wrapper events {note}"
        );
        let controller = controller.borrow();
        assert_eq!(
            controller.event,
            case["events"].as_array().unwrap().len(),
            "body events {note}"
        );
        assert_eq!(controller.real, case["real_after"].as_str().unwrap());
        check(
            &conditions.snapshot().unwrap(),
            &case["post"],
            Device::Cpu,
            &format!("post {note}"),
        );
        if !case["registry_counts"].is_null() {
            let Options::Dict(groups) = registry(&options) else {
                panic!("Own registry")
            };
            let counts = groups
                .borrow()
                .iter()
                .map(|(_, v)| sequence(v).len())
                .collect::<Vec<_>>();
            assert_eq!(json!(counts), case["registry_counts"], "registry {note}");
        }
    }
}
