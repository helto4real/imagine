use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::public_sample::{self, Arguments, Bindings},
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
fn text(value: &str) -> Options {
    Options::Atom(V::Text(value.into()))
}
fn name(value: &Options) -> &str {
    let Options::Atom(V::Text(value)) = value else {
        panic!("own token")
    };
    value
}
struct Ignored {
    events: Rc<RefCell<Vec<Value>>>,
    name: String,
}
impl Drop for Ignored {
    fn drop(&mut self) {
        self.events
            .borrow_mut()
            .push(json!(["drop-result", self.name]));
    }
}
struct Data<'a> {
    case: &'a Value,
    events: Rc<RefCell<Vec<Value>>>,
    replaced: Cell<bool>,
}
impl Data<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let index = self.events.borrow().len();
        self.events.borrow_mut().push(event);
        if self.case["failure"].as_u64() == Some(index as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own public sample failure".into(),
            });
        }
        Ok(())
    }
}
impl Bindings for Data<'_> {
    fn construct_guider(&self, model: &Options) -> Result<Options, Error> {
        self.emit(json!(["construct", name(model)]))?;
        Ok(text("guider"))
    }
    fn attribute(&self, object: &Options, attribute: &str) -> Result<Options, Error> {
        assert_eq!(name(object), "guider");
        self.emit(json!(["attribute", attribute]))?;
        Ok(text(attribute))
    }
    fn call(&self, function: &Options, args: &[Options]) -> Result<Options, Error> {
        let function = name(function);
        self.emit(json!([
            "call",
            function,
            args.iter().map(name).collect::<Vec<_>>(),
            self.replaced.get()
        ]))?;
        if function == "set_conds" && self.case["mode"] == "replace" {
            self.replaced.set(true);
        }
        Ok(if function == "sample" {
            text("output")
        } else {
            Options::Object(Rc::new(Ignored {
                events: self.events.clone(),
                name: function.into(),
            }))
        })
    }
}
#[test]
fn original_public_sampling_entry_preserves_guider_creation_live_method_reads_unused_parameters_ignored_result_release_and_failure_prefixes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-public-sample.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 16);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let bindings = Data {
            case,
            events: Rc::new(RefCell::new(vec![])),
            replaced: Cell::new(false),
        };
        let args = Arguments {
            model: text("model"),
            noise: text("noise"),
            positive: text("positive"),
            negative: text("negative"),
            cfg: text("cfg"),
            device: text("unused-device"),
            sampler: text("sampler"),
            sigmas: text("sigmas"),
            model_options: text("unused-options"),
            latent: text("latent"),
            mask: text("mask"),
            callback: text("callback"),
            disable: text("disable"),
            seed: text("seed"),
        };
        let (output, error) = match public_sample::sample(args, &bindings) {
            Ok(output) => (json!(name(&output)), Value::Null),
            Err(error) => match error.root() {
                Error::ReferenceException { class, message } => {
                    (Value::Null, json!([class, message]))
                }
                other => panic!("unbound {other:?}"),
            },
        };
        assert_eq!(output, case["output"], "output {index}");
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {index}"
        );
    }
}

#[test]
fn original_public_sample_errors_retain_all_parameters_unused_options_and_guider_until_traceback_release()
 {
    use std::{collections::BTreeMap, rc::Weak};
    struct Token;
    type Probes = Rc<RefCell<BTreeMap<String, Weak<Token>>>>;
    fn token(probes: &Probes, name: &str) -> Options {
        let token = Rc::new(Token);
        probes
            .borrow_mut()
            .insert(name.into(), Rc::downgrade(&token));
        Options::Object(token)
    }
    struct Provider {
        index: Cell<usize>,
        fail: usize,
        probes: Probes,
    }
    impl Provider {
        fn emit(&self) -> Result<(), Error> {
            let index = self.index.get();
            self.index.set(index + 1);
            if index == self.fail {
                Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "own public retained failure".into(),
                })
            } else {
                Ok(())
            }
        }
    }
    impl Bindings for Provider {
        fn construct_guider(&self, _: &Options) -> Result<Options, Error> {
            self.emit()?;
            Ok(token(&self.probes, "cfg_guider"))
        }
        fn attribute(&self, _: &Options, name: &str) -> Result<Options, Error> {
            self.emit()?;
            Ok(text(name))
        }
        fn call(&self, _: &Options, _: &[Options]) -> Result<Options, Error> {
            self.emit()?;
            Ok(Options::Atom(V::None))
        }
    }
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-public-sample-errors.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 7);
    for case in fixture["cases"].as_array().unwrap() {
        let probes = Probes::default();
        let args = Arguments {
            model: token(&probes, "model"),
            noise: token(&probes, "noise"),
            positive: token(&probes, "positive"),
            negative: token(&probes, "negative"),
            cfg: token(&probes, "cfg"),
            device: token(&probes, "device"),
            sampler: token(&probes, "sampler"),
            sigmas: token(&probes, "sigmas"),
            model_options: token(&probes, "model_options"),
            latent: token(&probes, "latent_image"),
            mask: token(&probes, "denoise_mask"),
            callback: token(&probes, "callback"),
            disable: token(&probes, "disable_pbar"),
            seed: token(&probes, "seed"),
        };
        let provider = Provider {
            index: Cell::new(0),
            fail: case["failure"].as_u64().unwrap() as usize,
            probes: probes.clone(),
        };
        let error = public_sample::sample(args, &provider)
            .err()
            .expect("original error");
        let exception = error.exception().unwrap();
        let mut names = exception.frames()[0].locals().names();
        names.sort();
        assert_eq!(json!(names), case["names"], "{case}");
        assert_eq!(json!(provider.index.get()), case["calls"]);
        let live = || {
            json!(
                probes
                    .borrow()
                    .iter()
                    .map(|(name, value)| (name, value.upgrade().is_some()))
                    .collect::<BTreeMap<_, _>>()
            )
        };
        assert_eq!(live(), case["retained"], "retained {case}");
        exception.clear_traceback();
        assert_eq!(live(), case["released"], "released {case}");
        match error.root() {
            Error::ReferenceException { class, message } => {
                assert_eq!(json!([class, message]), case["error"])
            }
            error => panic!("{error:?}"),
        }
    }
}
