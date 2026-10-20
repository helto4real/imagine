use super::{
    binding_tests::{check, reference_error, restore},
    device, kind, kitchen,
    shared_context_evaluation_tests::check_options,
    shared_context_state_tests::l,
};
use imagine_inference::{
    conditioning::{Condition, Kind as CK},
    context_windows::{
        self as c, BuiltinConditionAttributes, ConditionAttributes, Handler, LatentShape,
        SamplerWrapperBindings,
    },
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
    patching::{
        extensions::{Arguments, Original, WrapperExecutor},
        options::Options as O,
    },
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};
use tch::{Device, Tensor};
fn list(v: Vec<O>) -> O {
    O::List(l(v))
}
fn text(v: &str) -> O {
    O::Atom(V::Text(v.into()))
}
fn dict(v: Vec<(&str, O)>) -> O {
    O::dictionary(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn constant(v: O) -> O {
    O::Object(Rc::new(Condition::from_options(CK::Constant, v)))
}
struct Shared {
    case: Value,
    position: Cell<usize>,
    reads: Cell<usize>,
    forwards: Cell<usize>,
    infos: Cell<usize>,
    first: O,
    second: O,
    positive: O,
    negative: O,
    guider: O,
    sigmas: O,
    extra: O,
    callback: O,
    noise: V,
    token: O,
    own: O,
    h: Rc<Handler>,
    dev: Device,
    note: String,
}
impl Shared {
    fn mode(&self) -> &str {
        self.case["params"]["mode"].as_str().unwrap()
    }
    fn next(&self, tag: &str) -> Value {
        let i = self.position.get();
        let e = self.case["events"]
            .as_array()
            .unwrap()
            .get(i)
            .unwrap_or_else(|| panic!("{} unexpected {tag} at {i}", self.note))
            .clone();
        assert_eq!(e["tag"], tag, "{} event {i}", self.note);
        self.position.set(i + 1);
        e
    }
    fn error(&self, message: &str) -> Error {
        Error::ReferenceException {
            class: "ValueError",
            message: message.into(),
        }
    }
}
struct Bindings(Rc<Shared>);
impl ConditionAttributes for Bindings {
    fn cond(&self, v: &O) -> Result<O, Error> {
        BuiltinConditionAttributes.cond(v)
    }
}
impl SamplerWrapperBindings for Bindings {
    fn guider_conditions(&self, guider: &O) -> Result<O, Error> {
        let s = &self.0;
        assert!(guider.same_object(&s.guider), "{}", s.note);
        let n = s.reads.get() + 1;
        s.reads.set(n);
        let e = s.next("conditions");
        assert_eq!(e["read"], n, "{}", s.note);
        if s.mode() == "conditions-error-first" && n == 1
            || s.mode() == "conditions-error-second" && n == 2
        {
            return Err(s.error("own guider condition read failure"));
        }
        if s.mode() == "conditions-bad-first" && n == 1 {
            return Ok(O::Atom(V::Int(3)));
        }
        if s.mode() == "conditions-bad-second" && n == 2 {
            return Ok(list(vec![]));
        }
        if s.case["params"]["selection"] == "mutate-positive" && n == 2 {
            s.first.pop("positive")?;
        }
        Ok(if n == 1 {
            s.first.clone()
        } else {
            s.second.clone()
        })
    }
    fn info(&self, message: &str) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next("info");
        assert_eq!(e["message"], message, "{}", s.note);
        let n = s.infos.get() + 1;
        s.infos.set(n);
        if s.mode() == "info-error-first" && n == 1 || s.mode() == "info-error-second" && n == 2 {
            return Err(s.error("own sampler FreeNoise info failure"));
        }
        if s.mode() == "info-mutate" && n == 1 {
            s.h.settings.borrow_mut().length += 1;
        }
        Ok(())
    }
    fn freenoise_call(&self, noise: &V, conditions: &O, seed: &O) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next("freenoise");
        let O::List(v) = conditions else {
            panic!("own conditions")
        };
        let selected = &v.borrow()[0];
        assert_eq!(noise.same_object(&s.noise), e["same_noise"], "{}", s.note);
        check_options(selected, &e["selected"], &s.note);
        assert_eq!(
            selected.same_object(&s.positive),
            e["positive"],
            "{}",
            s.note
        );
        assert_eq!(
            selected.same_object(&s.negative),
            e["negative"],
            "{}",
            s.note
        );
        check(&seed.snapshot()?, &e["seed"], s.dev, &s.note);
        Ok(())
    }
}
#[test]
fn whole_context_sampler_wrapper_preserves_freenoise_descriptors_eager_defaults_seed_errors_forwarding_and_private_rng()
 {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _guard = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-sampler-wrapper.json"
    ))
    .unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["devices"], json!(["cpu", "cuda:0"]));
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 448);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = device(p["device"].as_str().unwrap());
        let multi = p["multi"] == true;
        let guides = p["guides"] == true;
        let mode = p["mode"].as_str().unwrap();
        let selection = p["selection"].as_str().unwrap();
        let call = p["call"].as_str().unwrap();
        let noise = restore(&case["input"], dev);
        let mut shapes = vec![vec![1, 1, 12, 2, 2]];
        if multi {
            shapes.push(vec![1, 1, 6, 1, 1]);
        }
        let shapes = list(
            shapes
                .into_iter()
                .map(|v| LatentShape::Size(Rc::new(v)).as_options())
                .collect(),
        );
        let positive_inner = dict(vec![]);
        let negative_inner = dict(vec![]);
        if multi {
            positive_inner
                .set("latent_shapes", constant(shapes.clone()))
                .unwrap();
            negative_inner
                .set("latent_shapes", constant(shapes))
                .unwrap();
        }
        if guides {
            positive_inner
                .set(
                    "guide_attention_entries",
                    constant(list(
                        [vec![2, 1, 1], vec![1, 2, 2]]
                            .into_iter()
                            .map(|v| {
                                dict(vec![(
                                    "latent_shape",
                                    O::Tuple(Rc::new(
                                        v.into_iter().map(|n| O::Atom(V::Int(n))).collect(),
                                    )),
                                )])
                            })
                            .collect(),
                    )),
                )
                .unwrap();
        }
        let positive = list(vec![dict(vec![("model_conds", positive_inner)])]);
        let negative = list(vec![dict(vec![("model_conds", negative_inner)])]);
        let first = dict(vec![
            ("positive", positive.clone()),
            ("negative", negative.clone()),
        ]);
        if matches!(selection, "negative" | "negative-none") {
            first.pop("positive").unwrap();
        }
        let first = if matches!(selection, "empty" | "different") {
            dict(vec![])
        } else {
            first
        };
        if selection == "positive-none" {
            first.set("positive", O::Atom(V::None)).unwrap();
        }
        if selection == "positive-empty" {
            first.set("positive", list(vec![])).unwrap();
        }
        if selection == "negative-none" {
            first.set("negative", O::Atom(V::None)).unwrap();
        }
        let second = if selection == "different" {
            dict(vec![("negative", positive.clone())])
        } else {
            first.clone()
        };
        let guider = O::Object(Rc::new("guider"));
        let sigmas = list(vec![text("sigmas")]);
        let callback = list(vec![text("callback")]);
        let own = list(vec![text("extra")]);
        let token = list(vec![text("return")]);
        let h = Rc::new(Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat));
        {
            let mut s = h.settings.borrow_mut();
            s.dim = 2;
            s.length = 4;
            s.overlap = 1;
            s.freenoise = p["enabled"] == true;
        }
        let options = dict(vec![("context_handler", O::Object(h.clone()))]);
        let seed = match p["seed"].as_str().unwrap() {
            "seven" => O::Atom(V::Int(7)),
            "zero" => O::Atom(V::Int(0)),
            "negative" => O::Atom(V::Int(-1)),
            "min" => O::Atom(V::Int(i64::MIN)),
            "max" => O::Atom(V::UInt(u64::MAX)),
            "bool" => O::Atom(V::Bool(true)),
            "float" => O::Atom(V::Float(1.5)),
            "none" => O::Atom(V::None),
            "list" => list(vec![]),
            "dict" => dict(vec![]),
            "tensor" => O::Atom(V::tensor(Tensor::from(7i64).to_device(dev))),
            "str" => text("7"),
            _ => panic!("own seed"),
        };
        let extra = dict(vec![("model_options", options.clone()), ("seed", seed)]);
        let extra = if mode == "extra-bad" {
            O::Atom(V::Int(3))
        } else {
            extra
        };
        match mode {
            "options-missing" => {
                extra.pop("model_options").unwrap();
            }
            "options-none" => {
                extra.set("model_options", O::Atom(V::None)).unwrap();
            }
            "options-bad" => {
                extra.set("model_options", O::Atom(V::Int(3))).unwrap();
            }
            "handler-missing" => {
                options.pop("context_handler").unwrap();
            }
            "handler-none" => {
                options.set("context_handler", O::Atom(V::None)).unwrap();
            }
            "handler-bad" => {
                options.set("context_handler", O::Atom(V::Int(3))).unwrap();
            }
            "seed-missing" => {
                extra.pop("seed").unwrap();
            }
            _ => {}
        }
        let mut args = vec![
            guider.clone(),
            sigmas.clone(),
            extra.clone(),
            callback.clone(),
            O::Atom(noise.clone()),
        ];
        let mut kwargs: Arguments<O> = vec![("own".into(), own.clone())];
        let names = ["guider", "sigmas", "extra_args", "callback", "noise"];
        match call {
            "keywords" => {
                kwargs.extend(
                    names
                        .iter()
                        .zip(&args)
                        .map(|(n, v)| (n.to_string(), v.clone())),
                );
                args.clear();
            }
            "mixed" => {
                kwargs.extend(
                    names
                        .iter()
                        .zip(&args)
                        .skip(2)
                        .map(|(n, v)| (n.to_string(), v.clone())),
                );
                args.truncate(2);
            }
            "extra" => args.extend([text("extra"), O::Atom(V::Int(7))]),
            "duplicate" => kwargs.push(("guider".into(), guider.clone())),
            "receiver" => kwargs.push(("executor".into(), O::Atom(V::Int(3)))),
            v if v.starts_with("missing-") => args.truncate(v[8..].parse().unwrap()),
            _ => {}
        }
        let shared = Rc::new(Shared {
            case: case.clone(),
            position: Cell::new(0),
            reads: Cell::new(0),
            forwards: Cell::new(0),
            infos: Cell::new(0),
            first,
            second,
            positive,
            negative,
            guider,
            sigmas,
            extra,
            callback,
            noise: noise.clone(),
            token: token.clone(),
            own,
            h: h.clone(),
            dev,
            note: format!("context sampler wrapper {index} {p}"),
        });
        let body: Original<O> = {
            let s = shared.clone();
            Rc::new(move |args, kwargs| {
                s.forwards.set(s.forwards.get() + 1);
                let e = s.next("executor");
                check(&args[4].snapshot()?, &e["noise"], s.dev, &s.note);
                assert_eq!(
                    json!([
                        args[0].same_object(&s.guider),
                        args[1].same_object(&s.sigmas),
                        args[2].same_object(&s.extra),
                        args[3].same_object(&s.callback),
                        args[4].same_object(&O::Atom(s.noise.clone()))
                    ]),
                    e["same"],
                    "{}",
                    s.note
                );
                check(
                    &V::Tuple(
                        args[5..]
                            .iter()
                            .map(O::snapshot)
                            .collect::<Result<_, _>>()?,
                    ),
                    &e["extra"],
                    s.dev,
                    &s.note,
                );
                assert_eq!(
                    json!(kwargs.iter().map(|(k, _)| k).collect::<Vec<_>>()),
                    e["keys"],
                    "{}",
                    s.note
                );
                assert_eq!(
                    kwargs
                        .iter()
                        .find(|(k, _)| k == "own")
                        .unwrap()
                        .1
                        .same_object(&s.own),
                    e["own_same"],
                    "{}",
                    s.note
                );
                if s.mode() == "executor-error" {
                    return Err(s.error("own sampler executor failure"));
                }
                Ok(s.token.clone())
            })
        };
        let wrapper =
            c::sampler_sample_wrapper(rt.native.clone(), Rc::new(Bindings(shared.clone())));
        let executor = WrapperExecutor::new(body, None, &[wrapper], 0);
        let cpu = native::global_rng_state(Device::Cpu).unwrap();
        let gpu = if dev != Device::Cpu {
            Some(native::global_rng_state(dev).unwrap())
        } else {
            None
        };
        let out = executor.execute(&args, &kwargs);
        match out {
            Ok(v) => {
                assert_eq!(case["ok"], true, "{}", shared.note);
                assert_eq!(
                    v.same_object(&token),
                    case["same_return"],
                    "{}",
                    shared.note
                );
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{} {e}", shared.note);
                reference_error(e, case);
            }
        }
        assert_eq!(
            shared.position.get(),
            case["events"].as_array().unwrap().len(),
            "{}",
            shared.note
        );
        assert_eq!(shared.reads.get(), case["reads"], "{}", shared.note);
        assert_eq!(shared.forwards.get(), case["forwards"], "{}", shared.note);
        assert_eq!(
            h.settings.borrow().length,
            case["length"],
            "{}",
            shared.note
        );
        check(&noise, &case["post"], dev, &shared.note);
        assert_eq!(case["global_rng_unchanged"], true);
        assert!(
            cpu.equal(&native::global_rng_state(Device::Cpu).unwrap()),
            "{}",
            shared.note
        );
        if let Some(gpu) = gpu {
            assert!(
                gpu.equal(&native::global_rng_state(dev).unwrap()),
                "{}",
                shared.note
            );
        }
        assert_eq!(
            noise.as_tensor().unwrap().kind(),
            kind(p["dtype"].as_str().unwrap())
        );
    }
}
