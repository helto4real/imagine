use super::binding_tests::reference_error;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        ejection::AutoPatcherEjector,
        extensions::OrderedRegistry,
        injections::{BoundInjections, InjectionBindings, InjectionLocals, InjectionState},
        options::Options as O,
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
struct Token {
    label: String,
}
#[derive(Clone)]
enum State {
    Legacy(Rc<InjectionState>),
    Source(Rc<imagine_inference::patching::model_patcher::PatcherState>),
}
struct RegistryKeys;
impl imagine_inference::patching::model_registry::RegistryBindings for RegistryKeys {
    fn empty_dictionary(&self) -> Result<O, Error> {
        Ok(O::dictionary(vec![]))
    }
}
fn name(value: &str) -> O {
    O::Atom(V::Text(value.into()))
}
impl State {
    fn new(source: bool) -> Self {
        if source {
            let state = Rc::new(
                imagine_inference::patching::model_patcher::PatcherState::uninitialized(
                    "ModelPatcher",
                ),
            );
            for field in ["callbacks", "injections"] {
                state.fields.set(field, O::dictionary(vec![]));
            }
            Self::Source(state)
        } else {
            Self::Legacy(Rc::new(InjectionState::new(Rc::new(RefCell::new(
                OrderedRegistry::default(),
            )))))
        }
    }
    fn legacy(&self) -> &Rc<InjectionState> {
        let Self::Legacy(state) = self else {
            panic!("own legacy ejection fixture")
        };
        state
    }
    fn flag(&self, field: &str) -> bool {
        match self {
            Self::Legacy(s) => {
                if field == "is_injected" {
                    s.is_injected.get()
                } else {
                    s.skip_injection.get()
                }
            }
            Self::Source(s) => {
                let O::Atom(V::Bool(v)) = s.fields.get(field).unwrap() else {
                    panic!("own bool")
                };
                v
            }
        }
    }
    fn set_flag(&self, field: &str, value: bool) {
        match self {
            Self::Legacy(s) => {
                if field == "is_injected" {
                    s.is_injected.set(value);
                } else {
                    s.skip_injection.set(value);
                }
            }
            Self::Source(s) => s.fields.set(field, O::Atom(V::Bool(value))),
        }
    }
    fn active(&self) -> bool {
        self.flag("is_injected")
    }
    fn skip(&self) -> bool {
        self.flag("skip_injection")
    }
    fn set_active(&self, value: bool) {
        self.set_flag("is_injected", value);
    }
    fn set_skip(&self, value: bool) {
        self.set_flag("skip_injection", value);
    }
    fn injections(&self) -> O {
        match self {
            Self::Legacy(s) => s.injections(),
            Self::Source(s) => s.fields.get("injections").unwrap(),
        }
    }
    fn replace_injections(&self, value: O) {
        match self {
            Self::Legacy(s) => s.replace_injections(value),
            Self::Source(s) => s.fields.set("injections", value),
        }
    }
    fn set_injections(&self, key: &str, value: O) -> Result<(), Error> {
        self.injections().set(key, value)
    }
    fn remove_injections(&self, key: &str) -> Result<(), Error> {
        drop(self.injections().pop(key)?);
        Ok(())
    }
    fn add_callback(&self, phase: &str, value: O) {
        match self {
            Self::Legacy(s) => s
                .callbacks
                .borrow_mut()
                .add(phase, Some("own"), Rc::new(value)),
            Self::Source(s) => s
                .registries
                .callbacks
                .add_with_key(&name(phase), &name("own"), value, &RegistryKeys)
                .unwrap(),
        }
    }
    fn clear_callbacks(&self) {
        match self {
            Self::Legacy(s) => {
                let old = std::mem::take(&mut *s.callbacks.borrow_mut());
                drop(old);
            }
            Self::Source(s) => s.fields.get("callbacks").unwrap().clear_dict().unwrap(),
        }
    }
    fn callback_entries(&self) -> Vec<Value> {
        if let Self::Source(s) = self {
            return encoded(&s.fields.get("callbacks").unwrap())["items"]
                .as_array()
                .unwrap()
                .clone();
        }
        let callbacks = self.legacy().callbacks.borrow();
        let mut entries = vec![];
        for phase in ["on_inject_model", "on_eject_model"] {
            let values = callbacks.list(phase, Some("own"));
            if !values.borrow().is_empty() {
                entries.push(json!([phase,{"type":"dict","items":[["own",{"type":"list","items":values.borrow().iter().map(|v|encoded(v)).collect::<Vec<_>>()}]]}]));
            }
        }
        entries
    }
    fn inject(&self, owner: &O, bindings: &Bindings) -> Result<(), Error> {
        match self {
            Self::Legacy(s) => s.inject_model(owner, bindings),
            Self::Source(s) => {
                imagine_inference::patching::injections::inject_model(s, owner, bindings)
            }
        }
    }
    fn eject(&self, owner: &O, bindings: &Bindings) -> Result<(), Error> {
        match self {
            Self::Legacy(s) => s.eject_model(owner, bindings),
            Self::Source(s) => {
                imagine_inference::patching::injections::eject_model(s, owner, bindings)
            }
        }
    }
}
type Group = RefCell<Vec<O>>;
struct Shared {
    case: Value,
    position: Cell<usize>,
    calls: Cell<usize>,
    callback_calls: Cell<usize>,
    next_injection: Cell<usize>,
    refs: RefCell<Vec<(String, Weak<Token>)>>,
    current_group: RefCell<Option<Weak<Group>>>,
    state: State,
    owner: O,
    note: String,
}
fn token(refs: &RefCell<Vec<(String, Weak<Token>)>>, label: String) -> O {
    let v = Rc::new(Token {
        label: label.clone(),
    });
    refs.borrow_mut().push((label, Rc::downgrade(&v)));
    O::Object(v)
}
fn label(v: &O) -> String {
    let O::Object(v) = v else {
        panic!("own injection or callback")
    };
    v.downcast_ref::<Token>().unwrap().label.clone()
}
fn list(v: Vec<O>) -> O {
    O::List(Rc::new(RefCell::new(v)))
}
fn encoded(v: &O) -> Value {
    match v {
        O::Object(_) => json!(label(v)),
        O::List(v) => {
            json!({"type":"list","items":v.borrow().iter().map(encoded).collect::<Vec<_>>()})
        }
        O::Tuple(v) => json!({"type":"tuple","items":v.iter().map(encoded).collect::<Vec<_>>()}),
        O::Dict(v) => {
            json!({"type":"dict","items":v.borrow().iter().map(|(k,v)|json!([k,encoded(v)])).collect::<Vec<_>>()})
        }
        O::Atom(V::None) => Value::Null,
        _ => panic!("unexpected own injection metadata"),
    }
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
    fn new_injection(&self) -> O {
        let i = self.next_injection.get();
        self.next_injection.set(i + 1);
        token(&self.refs, format!("i{i}"))
    }
    fn snapshot(&self) -> Value {
        let entries = self.state.callback_entries();
        json!({"injected":self.state.active(),"skip":self.state.skip(),"injections":encoded(&self.state.injections()),"callbacks":{"type":"dict","items":entries},"alive":self.refs.borrow().iter().map(|(k,v)|json!([k,v.upgrade().is_some()])).collect::<Vec<_>>()})
    }
    fn fail(&self, message: &str) -> Error {
        Error::ReferenceException {
            class: "ValueError",
            message: message.into(),
        }
    }
}
struct Bindings(Rc<Shared>);
impl imagine_inference::patching::model_registry::RegistryBindings for Bindings {
    fn empty_dictionary(&self) -> Result<O, Error> {
        Ok(O::dictionary(vec![]))
    }
}
impl imagine_inference::patching::injections::SourceInjectionBindings for Bindings {
    fn truth(&self, value: &O) -> Result<bool, Error> {
        let O::Atom(V::Bool(v)) = value else {
            panic!("own flag binding")
        };
        Ok(*v)
    }
    fn callback_kind(&self, value: &str) -> Result<O, Error> {
        Ok(name(value))
    }
}
impl Bindings {
    fn effect(&self, operation: &str, injection: &O, owner: &O) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next("effect");
        assert_eq!(e["operation"], operation, "{}", s.note);
        assert_eq!(e["label"], label(injection), "{}", s.note);
        assert_eq!(e["same_owner"], owner.same_object(&s.owner), "{}", s.note);
        let n = s.calls.get() + 1;
        s.calls.set(n);
        if s.mode() == "error-first" && n == 1 || s.mode() == "error-second" && n == 2 {
            return Err(s.fail("own injection effect failure"));
        }
        if n != 1 {
            return Ok(());
        }
        match s.mode() {
            "flag-false" => s.state.set_active(false),
            "flag-true-error" => {
                s.state.set_active(true);
                return Err(s.fail("own injection flag failure"));
            }
            "skip-true" => s.state.set_skip(true),
            "clear-list" => s
                .current_group
                .borrow()
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .borrow_mut()
                .clear(),
            "append-list" => s
                .current_group
                .borrow()
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .borrow_mut()
                .push(s.new_injection()),
            "replace-group" => s
                .state
                .set_injections("g0", list(vec![s.new_injection()]))?,
            "add-group" => s
                .state
                .set_injections("extra", list(vec![s.new_injection()]))?,
            "remove-group" => s.state.remove_injections("g0")?,
            "replace-root" => s.state.replace_injections(O::dictionary(vec![(
                "fresh".into(),
                list(vec![s.new_injection()]),
            )])),
            _ => {}
        }
        Ok(())
    }
}
impl InjectionBindings for Bindings {
    fn inject(&self, injection: &O, owner: &O) -> Result<(), Error> {
        self.effect("inject", injection, owner)
    }
    fn eject(&self, injection: &O, owner: &O) -> Result<(), Error> {
        self.effect("eject", injection, owner)
    }
    fn callback(&self, callback: &O, owner: &O) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next("effect");
        assert_eq!(e["operation"], "callback", "{}", s.note);
        assert_eq!(e["label"], label(callback), "{}", s.note);
        assert_eq!(e["same_owner"], owner.same_object(&s.owner), "{}", s.note);
        let n = s.callback_calls.get() + 1;
        s.callback_calls.set(n);
        if s.mode() == "callback-error" && n == 1 {
            return Err(s.fail("own injection callback failure"));
        }
        if s.mode() == "callback-clear" && n == 1 {
            s.state.clear_callbacks();
        }
        if s.mode() == "callback-flags" && n == 1 {
            s.state.set_active(!s.state.active());
            s.state.set_skip(!s.state.skip());
        }
        Ok(())
    }
    fn source_locals(&self, locals: &InjectionLocals, tag: &str) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next(tag);
        assert_eq!(
            e["method"],
            if locals.injecting {
                "inject_model"
            } else {
                "eject_model"
            },
            "{}",
            s.note
        );
        assert_eq!(
            e["group"],
            locals.group.as_ref().map(encoded).unwrap_or(Value::Null),
            "{} {tag}",
            s.note
        );
        assert_eq!(
            e["injection"],
            locals
                .injection
                .as_ref()
                .map(encoded)
                .unwrap_or(Value::Null),
            "{} {tag}",
            s.note
        );
        assert_eq!(
            e["callback"],
            locals
                .callback
                .as_ref()
                .map(|v| encoded(v))
                .unwrap_or(Value::Null),
            "{} {tag}",
            s.note
        );
        assert_eq!(e["state"], s.snapshot(), "{} {tag}", s.note);
        *s.current_group.borrow_mut() = match &locals.group {
            Some(O::List(v)) => Some(Rc::downgrade(v)),
            _ => None,
        };
        Ok(())
    }
}
fn outcome(out: Result<(), Error>, expected: &Value, s: &Shared) {
    match out {
        Ok(()) => assert_eq!(expected["ok"], true, "{}", s.note),
        Err(e) => {
            assert!(expected["error"].is_string(), "{} {e}", s.note);
            reference_error(e, expected);
        }
    }
    assert_eq!(
        s.snapshot(),
        expected["state"],
        "{} after caller handles result",
        s.note
    );
}
#[test]
fn whole_model_injection_and_ejection_preserve_live_maps_lists_callbacks_partial_flags_owners_errors_and_with_protocol()
 {
    replay(false);
}
#[test]
fn canonical_field_injections_preserve_live_fields_actual_registry_lists_partial_flags_owners_and_source_locals()
 {
    replay(true);
}
fn replay(source: bool) {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-injections.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["torch_python_imported"], false);
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 368);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        // AutoPatcherEjector's separate boolean owner contract is unchanged.
        // This selection covers the two canonical field injection bodies.
        if source && p["scope"] == true {
            continue;
        }
        let refs = RefCell::new(vec![]);
        let state = State::new(source);
        state.set_active(p["injected"].as_bool().unwrap());
        state.set_skip(p["skip"].as_bool().unwrap());
        let mut next = 0;
        for g in 0..p["groups"].as_u64().unwrap() {
            let values = (0..p["count"].as_u64().unwrap())
                .map(|_| {
                    let v = token(&refs, format!("i{next}"));
                    next += 1;
                    v
                })
                .collect::<Vec<_>>();
            let value = if p["layout"] == "tuple" {
                O::Tuple(Rc::new(values))
            } else {
                list(values)
            };
            state.set_injections(&format!("g{g}"), value).unwrap();
        }
        for phase in ["on_inject_model", "on_eject_model"] {
            for _ in 0..2 {
                let n = refs.borrow().len();
                state.add_callback(phase, token(&refs, format!("c{n}")));
            }
        }
        let shared = Rc::new(Shared {
            case: case.clone(),
            position: Cell::new(0),
            calls: Cell::new(0),
            callback_calls: Cell::new(0),
            next_injection: Cell::new(next),
            refs,
            current_group: RefCell::new(None),
            state: state.clone(),
            owner: O::Object(Rc::new("actual-model-owner")),
            note: format!("model injections {index} {p}"),
        });
        assert_eq!(shared.snapshot(), case["initial"], "{}", shared.note);
        let binding = Rc::new(Bindings(shared.clone()));
        if p["scope"] == true {
            let bound = Rc::new(BoundInjections {
                state: state.legacy().clone(),
                owner: shared.owner.clone(),
                bindings: binding.clone(),
            });
            let guard = AutoPatcherEjector::new(bound, p["scope_skip"].as_bool().unwrap());
            let out = guard.run(|| {
                let e = shared.next("body");
                assert_eq!(shared.snapshot(), e["state"], "{}", shared.note);
                if p["body_error"] == true {
                    Err(shared.fail("own ejection body failure"))
                } else {
                    Ok(())
                }
            });
            outcome(out, &case["outcomes"][0], &shared);
        } else {
            for (action, expected) in p["sequence"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["outcomes"].as_array().unwrap())
            {
                let out = if action == "inject" {
                    state.inject(&shared.owner, binding.as_ref())
                } else {
                    state.eject(&shared.owner, binding.as_ref())
                };
                outcome(out, expected, &shared);
            }
        }
        assert_eq!(
            shared.position.get(),
            case["events"].as_array().unwrap().len(),
            "{}",
            shared.note
        );
    }
}
