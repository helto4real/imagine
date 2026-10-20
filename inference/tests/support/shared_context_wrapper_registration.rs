use super::{binding_tests::reference_error, kitchen};
use imagine_inference::{
    context_windows::{
        ConditionAttributes, ContextWrapperModel, ContextWrappers, SamplerWrapperBindings,
    },
    native::Error,
    patching::{extensions::Wrapper, options::Options as O},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
struct Bindings;
impl ConditionAttributes for Bindings {
    fn cond(&self, _: &O) -> Result<O, Error> {
        panic!("registration does not read conditions")
    }
}
impl SamplerWrapperBindings for Bindings {
    fn guider_conditions(&self, _: &O) -> Result<O, Error> {
        panic!("registration does not call guider")
    }
    fn info(&self, _: &str) -> Result<(), Error> {
        panic!("registration does not execute FreeNoise")
    }
}
struct Shared {
    case: Value,
    functions: RefCell<Vec<(String, Wrapper<O>)>>,
    note: String,
}
struct Model {
    shared: Rc<Shared>,
    index: usize,
}
impl ContextWrapperModel for Model {
    fn add_wrapper_with_key(
        &self,
        kind: &str,
        key: &str,
        wrapper: Wrapper<O>,
    ) -> Result<(), Error> {
        let s = &self.shared;
        let mut functions = s.functions.borrow_mut();
        let n = functions.len();
        let e = &s.case["events"][n];
        assert_eq!(json!(self.index), e["model"], "{}", s.note);
        assert_eq!(kind, e["kind"], "{}", s.note);
        assert_eq!(key, e["key"], "{}", s.note);
        assert_eq!(
            json!(
                functions
                    .iter()
                    .map(|(_, v)| Rc::ptr_eq(v, &wrapper))
                    .collect::<Vec<_>>()
            ),
            e["same_previous"],
            "{}",
            s.note
        );
        let name = if kind == "prepare_sampling" {
            "_prepare_sampling_wrapper"
        } else {
            "_sampler_sample_wrapper"
        };
        assert_eq!(name, e["name"], "{}", s.note);
        // First observation establishes the module's canonical owner; all
        // later registrations of that function must pass this same object.
        let same = functions
            .iter()
            .find(|(k, _)| k == kind)
            .is_none_or(|(_, v)| Rc::ptr_eq(v, &wrapper));
        assert_eq!(same, e["same_canonical"], "{}", s.note);
        functions.push((kind.into(), wrapper));
        if s.case["params"]["mode"] == "error-first" && n == 0
            || s.case["params"]["mode"] == "error-second" && n == 1
        {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own wrapper registration failure".into(),
            });
        }
        Ok(())
    }
}
#[test]
fn context_wrapper_registration_preserves_call_keys_canonical_function_identity_models_and_partial_failures()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-wrapper-registration.json"
    ))
    .unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["torch_python_imported"], false);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 180);
    let module = ContextWrappers::new(kitchen(), Rc::new(Bindings));
    for (index, case) in cases.iter().enumerate() {
        let shared = Rc::new(Shared {
            case: case.clone(),
            functions: RefCell::new(vec![]),
            note: format!("context registration {index}"),
        });
        let n = case["params"]["models"].as_u64().unwrap() as usize;
        let owners = (0..n)
            .map(|index| Model {
                shared: shared.clone(),
                index,
            })
            .collect::<Vec<_>>();
        let out = (|| {
            for (i, action) in case["params"]["sequence"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                if action == "prepare" {
                    module.create_prepare_sampling_wrapper(&owners[i % n])?;
                } else {
                    module.create_sampler_sample_wrapper(&owners[i % n])?;
                }
            }
            Ok::<_, Error>(())
        })();
        match out {
            Ok(()) => {
                assert_eq!(case["ok"], true, "{}", shared.note);
                assert_eq!(
                    case["returns_none"],
                    json!(vec![
                        true;
                        case["params"]["sequence"].as_array().unwrap().len()
                    ])
                );
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{} {e}", shared.note);
                reference_error(e, case);
            }
        }
        assert_eq!(
            shared.functions.borrow().len(),
            case["events"].as_array().unwrap().len(),
            "{}",
            shared.note
        );
    }
}
