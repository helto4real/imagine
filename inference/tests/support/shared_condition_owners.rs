use imagine_inference::{
    conditioning::{Condition, Kind},
    execution_value::Value as V,
    patching::options::Options,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
struct Token {
    label: Cell<usize>,
}
fn token(refs: &mut Vec<Weak<Token>>) -> Options {
    let t = Rc::new(Token {
        label: Cell::new(refs.len()),
    });
    refs.push(Rc::downgrade(&t));
    Options::Object(t)
}
fn payload(style: &str, refs: &mut Vec<Weak<Token>>) -> Options {
    match style {
        "token" => token(refs),
        "list" => Options::List(Rc::new(RefCell::new(vec![token(refs), token(refs)]))),
        _ => Options::dictionary(vec![
            ("inner".into(), token(refs)),
            ("other".into(), token(refs)),
        ]),
    }
}
fn as_token(v: &Options) -> &Token {
    let Options::Object(v) = v else {
        panic!("own token")
    };
    v.downcast_ref::<Token>().unwrap()
}
fn encode(v: &Options) -> Value {
    match v {
        Options::Object(_) => json!(["token", as_token(v).label.get()]),
        Options::List(v) => json!(["list", v.borrow().iter().map(encode).collect::<Vec<_>>()]),
        Options::Dict(v) => json!([
            "dict",
            v.borrow()
                .iter()
                .map(|(k, v)| json!([k, encode(v)]))
                .collect::<Vec<_>>()
        ]),
        _ => panic!("own payload"),
    }
}
fn class(kind: Kind) -> &'static str {
    match kind {
        Kind::Regular => "CONDRegular",
        Kind::NoiseShape => "CONDNoiseShape",
        Kind::CrossAttention => "CONDCrossAttn",
        Kind::Constant => "CONDConstant",
        Kind::List => "CONDList",
    }
}
fn kind(name: &str) -> Kind {
    [
        Kind::Regular,
        Kind::NoiseShape,
        Kind::CrossAttention,
        Kind::Constant,
        Kind::List,
    ]
    .into_iter()
    .find(|k| class(*k) == name)
    .unwrap()
}
fn snapshot(holders: &[(String, Condition)], refs: &[Weak<Token>]) -> Value {
    let values = holders
        .iter()
        .map(|(_, v)| v.source_value())
        .collect::<Vec<_>>();
    let labels = holders
        .iter()
        .zip(&values)
        .map(|((key, v), value)| json!([key, class(v.kind), encode(value)]))
        .collect::<Vec<_>>();
    let same_payloads = values
        .iter()
        .map(|a| values.iter().map(|b| a.same_object(b)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    drop(values);
    json!({"holders":labels,"same_wrappers":holders.iter().map(|(_,a)|holders.iter().map(|(_,b)|a.same_object(b)).collect::<Vec<_>>()).collect::<Vec<_>>(),"same_payloads":same_payloads,"live":refs.iter().map(|v|v.upgrade().is_some()).collect::<Vec<_>>()})
}
#[test]
fn condition_aliases_copies_assignments_and_mutable_payloads_preserve_original_wrapper_and_value_owners()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-condition-owners.json")).unwrap();
    assert_eq!(fixture["torch_python_imported"], false);
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 15);
    for case in cases {
        let k = kind(case["name"].as_str().unwrap());
        let style = case["style"].as_str().unwrap();
        let mut refs = vec![];
        let a = Condition::from_options(k, payload(style, &mut refs));
        let b = a.clone();
        let c = a.copy_with_options(a.source_value());
        let mut holders = vec![("a".into(), a), ("b".into(), b), ("c".into(), c)];
        assert_eq!(snapshot(&holders, &refs), case["initial"]);
        for (index, event) in case["events"].as_array().unwrap().iter().enumerate() {
            let action = &event["action"];
            let target = action["target"].as_str().unwrap();
            let find = |name: &str| holders.iter().position(|(k, _)| k == name).unwrap();
            match action["op"].as_str().unwrap() {
                "set" => holders[find(target)].1.set_value(payload(style, &mut refs)),
                "copy" | "alias" => {
                    let source = &holders[find(action["source"].as_str().unwrap())].1;
                    let value = if action["op"] == "copy" {
                        source.copy_with_options(source.source_value())
                    } else {
                        source.clone()
                    };
                    holders.push((target.into(), value));
                }
                "drop" => {
                    holders.remove(find(target));
                }
                "mutate" => match holders[find(target)].1.source_value() {
                    Options::Object(v) => {
                        let t = v.downcast_ref::<Token>().unwrap();
                        t.label.set(t.label.get() + 1000);
                    }
                    Options::List(v) => v.borrow_mut()[0] = token(&mut refs),
                    d @ Options::Dict(_) => d.set("inner", token(&mut refs)).unwrap(),
                    _ => unreachable!(),
                },
                _ => panic!("own action"),
            }
            assert_eq!(
                snapshot(&holders, &refs),
                event["post"],
                "{} {style} {index} {action}",
                class(k)
            );
        }
        assert!(refs.iter().all(|v| v.upgrade().is_none()));
    }
}
struct Finalizer {
    call: Box<dyn Fn()>,
}
impl Drop for Finalizer {
    fn drop(&mut self) {
        (self.call)();
    }
}
#[test]
fn condition_field_assignment_is_visible_before_reentrant_finalizer_and_allows_its_replacement() {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-condition-owners.json")).unwrap();
    assert_eq!(f["reentrant"].as_array().unwrap().len(), 5);
    for case in f["reentrant"].as_array().unwrap() {
        let cond = Condition::new(kind(case["name"].as_str().unwrap()), V::None);
        let events = Rc::new(RefCell::new(vec![]));
        let captured = cond.clone();
        let seen = events.clone();
        let text = |c: &Condition| {
            let V::Text(t) = c.value().unwrap() else {
                panic!("field replacement not committed")
            };
            t
        };
        cond.set_value(Options::Object(Rc::new(Finalizer {
            call: Box::new(move || {
                seen.borrow_mut().push(json!(["before", text(&captured)]));
                captured.set_value(Options::Atom(V::Text("from-finalizer".into())));
                seen.borrow_mut().push(json!(["after", text(&captured)]));
            }),
        })));
        let alias = cond.clone();
        alias.set_value(Options::Atom(V::Text("replacement".into())));
        assert_eq!(json!(events.borrow().clone()), case["events"]);
        assert_eq!(json!(text(&alias)), case["output"]);
    }
}
