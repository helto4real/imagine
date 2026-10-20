use imagine_inference::{native::Error, source_exception::Frame};
use serde_json::{Value, json};
use std::rc::Rc;

#[test]
fn original_exception_context_cause_properties_identity_cycle_breaking_and_finally_root_retention()
{
    use imagine_inference::source_exception::{Exception, ExceptionLink, finally};
    fn errors() -> Vec<Rc<Exception>> {
        (0..3)
            .map(|i| {
                Error::ReferenceException {
                    class: "ValueError",
                    message: format!("own {i}"),
                }
                .into_exception()
            })
            .collect()
    }
    fn state(errors: &[Rc<Exception>]) -> Value {
        let index = |value: Option<Rc<Exception>>| {
            value.map(|value| {
                errors
                    .iter()
                    .position(|other| Rc::ptr_eq(other, &value))
                    .unwrap()
            })
        };
        json!(errors.iter().map(|error|json!({"context":index(error.context()),"cause":index(error.cause()),"suppress":error.suppress_context()})).collect::<Vec<_>>())
    }
    fn clear(errors: &[Rc<Exception>]) {
        for error in errors {
            error.set_context(ExceptionLink::None).unwrap();
            error.set_cause(ExceptionLink::None).unwrap();
        }
    }
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-exception-context.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 60);
    for case in fixture["cases"].as_array().unwrap() {
        let errors = errors();
        for (i, link) in case["links"].as_array().unwrap().iter().enumerate() {
            errors[i]
                .set_context(
                    link.as_u64()
                        .map(|i| ExceptionLink::Exception(errors[i as usize].clone()))
                        .unwrap_or(ExceptionLink::None),
                )
                .unwrap();
        }
        let raised = errors[case["raised"].as_u64().unwrap() as usize].clone();
        if let Some(current) = case["current"].as_u64() {
            let result = Error::Retained(raised.clone())
                .during_handling(Error::Retained(errors[current as usize].clone()));
            assert!(Rc::ptr_eq(&raised, &result.exception().unwrap()));
        }
        assert_eq!(state(&errors), case["state"], "{case}");
        clear(&errors);
    }
    assert_eq!(fixture["properties"].as_array().unwrap().len(), 21);
    for case in fixture["properties"].as_array().unwrap() {
        let errors = errors();
        let value = &case["value"];
        let link = || {
            if value == "none" {
                ExceptionLink::None
            } else if let Some(i) = value.as_u64() {
                ExceptionLink::Exception(errors[i as usize].clone())
            } else {
                ExceptionLink::Other
            }
        };
        let result = match case["name"].as_str().unwrap() {
            "__cause__" => errors[0].set_cause(link()),
            "__context__" => errors[0].set_context(link()),
            "__suppress_context__" => errors[0].set_suppress_context(value.as_bool()),
            _ => unreachable!(),
        };
        let error = match result {
            Ok(()) => Value::Null,
            Err(Error::ReferenceException { class, message }) => json!([class, message]),
            Err(error) => panic!("{error:?}"),
        };
        assert_eq!(error, case["error"], "{case}");
        assert_eq!(state(&errors), case["state"], "{case}");
        clear(&errors);
    }
    let token = Rc::new(());
    let weak = Rc::downgrade(&token);
    let frame = Frame::new("body");
    frame.set("buffer", token);
    frame.finish();
    let body = Error::ReferenceException {
        class: "ValueError",
        message: "body".into(),
    }
    .retain_frame(frame);
    let error = finally::<()>(Err(body), || {
        Err(Error::ReferenceException {
            class: "RuntimeError",
            message: "cleanup".into(),
        })
    })
    .unwrap_err();
    assert!(weak.upgrade().is_some());
    let exception = error.exception().unwrap();
    exception.clear_traceback();
    assert!(weak.upgrade().is_some());
    exception.set_context(ExceptionLink::None).unwrap();
    assert!(weak.upgrade().is_none());
}

#[test]
fn original_traceback_frame_clear_and_exposed_dictionary_preserve_independent_roots() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-exception-roots.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let token = Rc::new(());
        let weak = Rc::downgrade(&token);
        let frame = Frame::new("inner");
        frame.set("value", token.clone());
        frame.set("owned", token);
        frame.finish();
        let mut error = Some(
            Error::ReferenceException {
                class: "ValueError",
                message: "own retained failure".into(),
            }
            .retain_frame(frame.clone()),
        );
        let mut dictionary = case["cache"].as_bool().unwrap().then(|| frame.locals());
        if let Some(dictionary) = &dictionary {
            dictionary.set("external", Rc::new("own external entry"));
        }
        let mut events = vec![json!(["caught", weak.upgrade().is_some()])];
        let kept = case["external"].as_bool().unwrap().then(|| frame.clone());
        if case["clear"] == "frame" {
            frame.clear().unwrap();
        } else {
            error
                .as_ref()
                .unwrap()
                .exception()
                .unwrap()
                .clear_traceback();
        }
        drop(frame);
        events.push(json!(["cleared", weak.upgrade().is_some()]));
        if let Some(kept) = &kept {
            let mut names = kept.locals().names();
            names.sort();
            events.push(json!(["locals", names, weak.upgrade().is_some()]));
        }
        drop(kept);
        events.push(json!(["frame-dropped", weak.upgrade().is_some()]));
        drop(error.take());
        events.push(json!(["exception-dropped", weak.upgrade().is_some()]));
        drop(dictionary.take());
        events.push(json!(["dictionary-dropped", weak.upgrade().is_some()]));
        assert_eq!(json!(events), case["snapshots"], "{case}");
    }
    let frame = Frame::new("active");
    let error = frame.clear().unwrap_err();
    if let Error::ReferenceException { class, message } = error {
        assert_eq!(json!([class, message]), fixture["active_error"]);
    } else {
        panic!("missing original active-frame exception");
    }
}

#[test]
fn retained_errors_preserve_identity_nested_frame_order_oom_dispatch_and_reentrant_release() {
    use imagine_inference::memory::exceptions::{
        NativeOomBindings, is_oom, native_runtime_error, raise_non_oom,
    };
    let token = Rc::new(());
    let weak = Rc::downgrade(&token);
    let inner = Frame::new("inner");
    inner.set("value", token);
    inner.finish();
    let error = Error::AcceleratorException {
        code: 2,
        message: "own allocation failure".into(),
    }
    .retain_frame(inner);
    let identity = error.exception().unwrap();
    let outer = Frame::new("outer");
    outer.finish();
    let error = error.retain_frame(outer);
    assert!(Rc::ptr_eq(&identity, &error.exception().unwrap()));
    assert_eq!(
        identity
            .frames()
            .iter()
            .map(|f| f.function.as_str())
            .collect::<Vec<_>>(),
        ["outer", "inner"]
    );
    let mut discards = 0;
    assert!(
        is_oom(
            &error,
            &mut NativeOomBindings(|| {
                discards += 1;
                Ok(())
            })
        )
        .unwrap()
    );
    assert_eq!(discards, 1);
    assert!(native_runtime_error(&error));
    drop(error);
    assert!(weak.upgrade().is_some());
    identity.clear_traceback();
    assert!(weak.upgrade().is_none());
    let frame = Frame::new("non_oom");
    frame.finish();
    let error = Error::ReferenceException {
        class: "ValueError",
        message: "own".into(),
    }
    .retain_frame(frame);
    let identity = error.exception().unwrap();
    let returned = raise_non_oom(
        error,
        &mut NativeOomBindings(|| panic!("non-OOM must not discard")),
    )
    .unwrap_err();
    assert!(Rc::ptr_eq(&identity, &returned.exception().unwrap()));
    assert_eq!(returned.to_string(), "own");
    struct Reenter(Rc<imagine_inference::source_exception::Frame>);
    impl Drop for Reenter {
        fn drop(&mut self) {
            self.0.set("replacement", Rc::new(7));
        }
    }
    let frame = Frame::new("reentrant");
    frame.set("old", Rc::new(Reenter(frame.clone())));
    frame.finish();
    frame.clear().unwrap();
    assert!(frame.locals().get("replacement").is_some());
}

#[test]
fn original_churn_model_and_callback_failures_retain_exact_live_tensor_frames_until_traceback_release()
 {
    use imagine_inference::{
        execution_value::Value as V,
        models::qwen_image21::{EulerSettings, EulerStep},
        native::WeakStorage,
        sampling::solvers::{self, ChurnSolver, Execution, NoiseKind, Prediction},
    };
    use std::collections::BTreeMap;
    use tch::{Kind, Tensor};
    struct Exec {
        failure: String,
        index: usize,
        calls: [usize; 2],
        noise: imagine_inference::native::Generator,
    }
    impl Exec {
        fn hit(&mut self, which: usize, name: &str) -> Result<(), Error> {
            let index = self.calls[which];
            self.calls[which] += 1;
            if self.failure == name && index == self.index {
                Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "own retained sampler failure".into(),
                })
            } else {
                Ok(())
            }
        }
    }
    impl Execution for Exec {
        fn denoise(&mut self, x: &Tensor, sigma: &Tensor, _: bool) -> Result<Prediction, Error> {
            self.hit(0, "model")?;
            Ok(Prediction {
                denoised: x
                    .f_mul_scalar(0.125)?
                    .f_add(&sigma.f_reshape([2, 1, 1, 1])?.f_mul_scalar(0.25)?)?,
                uncond_denoised: None,
                cond_denoised: None,
            })
        }
        fn callback(&mut self, _: EulerStep<'_>) -> Result<(), Error> {
            self.hit(1, "callback")
        }
        fn noise(
            &mut self,
            x: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: NoiseKind,
        ) -> Result<Tensor, Error> {
            // Heun++2 allocates eps even with zero churn. Its zero contribution
            // still affects ownership; a real random draw is needed here.
            self.noise.randn(&x.size(), x.kind())
        }
    }
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-sampler-error-roots.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 64);
    let native = super::kitchen();
    for case in fixture["cases"].as_array().unwrap() {
        let device = super::device(case["device"].as_str().unwrap());
        let kind = if case["kind"] == "torch.float32" {
            Kind::Float
        } else {
            Kind::BFloat16
        };
        let input: Vec<f32> = (0..12)
            .map(|i| ((i + 3) % 29) as f32 * 0.03125 - 0.40625)
            .collect();
        let x = Tensor::from_slice(&input)
            .reshape([2, 1, 2, 3])
            .to_kind(kind)
            .to_device(device);
        let sigmas = Tensor::from_slice(&[0.9_f32, 0.6, 0.3, 0.1, 0.]).to_device(device);
        let mut exec = Exec {
            failure: case["failure"].as_str().unwrap().into(),
            index: case["index"].as_u64().unwrap() as usize,
            calls: [0, 0],
            noise: native.generator(device, 17).unwrap(),
        };
        let solver = match case["name"].as_str().unwrap() {
            "euler" => ChurnSolver::Euler,
            "heun" => ChurnSolver::Heun,
            "dpm_2" => ChurnSolver::Dpm2,
            "heunpp2" => ChurnSolver::HeunPP2,
            _ => unreachable!(),
        };
        let error =
            solvers::churn(&mut exec, x, &sigmas, EulerSettings::default(), solver).unwrap_err();
        assert_eq!(json!(exec.calls), case["calls"], "{case}");
        let exception = error.exception().expect("escaped source errors own frames");
        let frames = exception.frames();
        assert_eq!(
            frames
                .iter()
                .map(|f| f.function.clone())
                .collect::<Vec<_>>(),
            [
                "decorate_context".to_owned(),
                format!("sample_{}", case["name"].as_str().unwrap())
            ]
        );
        let frame = frames[1].locals();
        let mut probes = BTreeMap::new();
        let fields = case["fields"].as_object().unwrap();
        let mut names = frame.names();
        names.sort();
        assert_eq!(
            names,
            fields.keys().cloned().collect::<Vec<_>>(),
            "source tensor locals {case}"
        );
        for (name, expected) in fields {
            let root = frame.get(name).unwrap();
            let tensor = root.downcast_ref::<Tensor>().unwrap();
            super::binding_tests::check(&V::tensor(tensor.shallow_clone()), expected, device, name);
            probes.insert(name.clone(), WeakStorage::new(tensor).unwrap());
        }
        let live = |probes: &BTreeMap<String, WeakStorage>| -> Value {
            json!(
                probes
                    .iter()
                    .map(|(name, value)| (name, !value.expired()))
                    .collect::<BTreeMap<_, _>>()
            )
        };
        assert_eq!(live(&probes), case["retained"], "retained {case}");
        drop(frame);
        drop(frames);
        exception.clear_traceback();
        assert_eq!(live(&probes), case["released"], "released {case}");
        assert_eq!(error.to_string(), case["error"][1]);
    }
}
