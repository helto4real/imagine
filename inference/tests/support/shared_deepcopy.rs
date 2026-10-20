use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        deepcopy::{CopyMemo, DeepCopyBindings, deepcopy_list_dict},
        options::{GeneralMapping, Options, SourceText},
    },
};
use serde_json::{Value, json};
use std::{cell::Cell, cell::RefCell, rc::Rc};

struct Token(String);
fn same(a: &Options, b: &Options) -> bool {
    a.same_object(b)
        || match (a, b) {
            (Options::Atom(V::Text(a)), Options::Atom(V::Text(b))) => a == b,
            (Options::Atom(V::Int(a)), Options::Atom(V::Int(b))) => a == b,
            (Options::Atom(V::None), Options::Atom(V::None)) => true,
            _ => false,
        }
}
fn restore(definitions: &[Value], general: bool) -> Vec<Options> {
    let keys = super::own_mapping_keys::OwnMappingKeys::seeded();
    let mut values = definitions
        .iter()
        .map(|definition| match definition[0].as_str().unwrap() {
            "text" if general => SourceText::from_utf8(definition[1].as_str().unwrap()),
            "text" => Options::Atom(V::Text(definition[1].as_str().unwrap().into())),
            "object" => Options::Object(Rc::new(Token(definition[1].as_str().unwrap().into()))),
            "int" => Options::Atom(V::Int(definition[1].as_i64().unwrap())),
            "list" => Options::List(Rc::new(RefCell::new(vec![]))),
            "dict" if general => keys.empty(),
            "dict" => Options::dictionary(vec![]),
            "tuple" | "none" => Options::Atom(V::None),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    for (index, definition) in definitions.iter().enumerate() {
        if definition[0] == "tuple" {
            values[index] = Options::Tuple(Rc::new(
                definition[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| values[v.as_u64().unwrap() as usize].clone())
                    .collect(),
            ));
        }
    }
    for (index, definition) in definitions.iter().enumerate() {
        match &values[index] {
            _ if GeneralMapping::get(&values[index]).is_some() => {
                for pair in definition[1].as_array().unwrap() {
                    values[index]
                        .set_item(
                            values[pair[0].as_u64().unwrap() as usize].clone(),
                            values[pair[1].as_u64().unwrap() as usize].clone(),
                        )
                        .unwrap();
                }
            }
            Options::List(list) => list.borrow_mut().extend(
                definition[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| values[v.as_u64().unwrap() as usize].clone()),
            ),
            Options::Dict(_) => {
                for pair in definition[1].as_array().unwrap() {
                    let Options::Atom(V::Text(key)) = &values[pair[0].as_u64().unwrap() as usize]
                    else {
                        unreachable!()
                    };
                    values[index]
                        .set(key, values[pair[1].as_u64().unwrap() as usize].clone())
                        .unwrap();
                }
            }
            _ => (),
        }
    }
    values
}
struct Bindings {
    values: Vec<Options>,
    active: Cell<usize>,
    // Own execution-context contract test, not a new engine recursion policy.
    maximum: usize,
    recursion_message: String,
}
impl DeepCopyBindings for Bindings {
    fn object_id(&self, value: &Options) -> Result<u64, Error> {
        Ok(self.values.iter().position(|v| same(v, value)).unwrap() as u64)
    }
    fn key_object(&self, _: &Options, key: &str) -> Result<Options, Error> {
        Ok(self
            .values
            .iter()
            .find(|value| matches!(value, Options::Atom(V::Text(text)) if text == key))
            .unwrap()
            .clone())
    }
    fn enter_recursion(&self) -> Result<(), Error> {
        if self.active.get() == self.maximum {
            return Err(Error::ReferenceException {
                class: "RecursionError",
                message: self.recursion_message.clone(),
            });
        }
        self.active.set(self.active.get() + 1);
        Ok(())
    }
    fn leave_recursion(&self) {
        self.active.set(self.active.get() - 1);
    }
}
fn index(objects: &mut Vec<Options>, value: &Options) -> usize {
    if let Some(position) = objects.iter().position(|v| same(v, value)) {
        position
    } else {
        objects.push(value.clone());
        objects.len() - 1
    }
}
fn snapshot(values: &[Options], output: Option<&Options>, memo: &CopyMemo) -> Value {
    let mut objects = values.to_vec();
    let output = output.map(|value| index(&mut objects, value));
    let memo = memo
        .entries()
        .iter()
        .map(|(id, value)| json!([id, index(&mut objects, value)]))
        .collect::<Vec<_>>();
    let mut encoded = vec![];
    let mut position = 0;
    while position < objects.len() {
        let value = objects[position].clone();
        let entry = match &value {
            _ if GeneralMapping::get(&value).is_some() => json!([
                "dict",
                GeneralMapping::get(&value)
                    .unwrap()
                    .dictionary
                    .entries()
                    .iter()
                    .map(|(k, v)| json!([index(&mut objects, k), index(&mut objects, v)]))
                    .collect::<Vec<_>>()
            ]),
            _ if SourceText::get(&value).is_some() => {
                json!(["text", SourceText::get(&value).unwrap().utf8().unwrap()])
            }
            Options::List(list) => json!([
                "list",
                list.borrow()
                    .iter()
                    .map(|v| index(&mut objects, v))
                    .collect::<Vec<_>>()
            ]),
            Options::Tuple(list) => json!([
                "tuple",
                list.iter()
                    .map(|v| index(&mut objects, v))
                    .collect::<Vec<_>>()
            ]),
            Options::Dict(dict) => json!([
                "dict",
                dict.borrow()
                    .iter()
                    .map(|(k, v)| json!([
                        index(&mut objects, &Options::Atom(V::Text(k.clone()))),
                        index(&mut objects, v)
                    ]))
                    .collect::<Vec<_>>()
            ]),
            Options::Object(object) => json!(["object", object.downcast_ref::<Token>().unwrap().0]),
            Options::Atom(V::Text(text)) => json!(["text", text]),
            Options::Atom(V::Int(value)) => json!(["int", value]),
            Options::Atom(V::None) => json!(["none"]),
            _ => unreachable!(),
        };
        encoded.push(entry);
        position += 1;
    }
    json!({"output":output,"memo":memo,"objects":encoded})
}
fn run(cyclic: bool, general: bool) {
    run_fixture(
        include_str!("../../fixtures/shared-deepcopy.json"),
        cyclic,
        general,
        if cyclic { 53 } else { 178 },
    );
}
fn run_fixture(source: &str, cyclic: bool, general: bool, count: usize) {
    let fixture: Value = serde_json::from_str(source).unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    let definitions = fixture["definitions"].as_array().unwrap();
    let cases = fixture[if cyclic { "recursive" } else { "cases" }]
        .as_array()
        .unwrap();
    assert_eq!(cases.len(), count);
    for (number, case) in cases.iter().enumerate() {
        let values = restore(definitions, general);
        let bindings = Bindings {
            values: values.clone(),
            active: Cell::new(0),
            maximum: if cyclic { 128 } else { 4096 },
            recursion_message: case["message"].as_str().unwrap_or("").into(),
        };
        let mut memo = CopyMemo::default();
        if let Some(seed) = case["seed"].as_array() {
            for pair in seed {
                memo.set(
                    pair[0].as_u64().unwrap(),
                    values[pair[1].as_u64().unwrap() as usize].clone(),
                );
            }
        }
        let result = deepcopy_list_dict(
            &values[case["root"].as_u64().unwrap() as usize],
            (!case["seed"].is_null()).then_some(&mut memo),
            &bindings,
        );
        assert_eq!(bindings.active.get(), 0, "balanced context {number}");
        let outcome = match &result {
            Ok(_) => json!({"ok":true}),
            Err(Error::ReferenceException { class, message }) => {
                json!({"error":class,"message":message})
            }
            Err(error) => panic!("unbound case {number}: {error:?}"),
        };
        let expected = if case.get("ok").is_some() {
            json!({"ok":true})
        } else {
            json!({"error":case["error"],"message":case["message"]})
        };
        assert_eq!(outcome, expected, "case {number}");
        assert_eq!(
            snapshot(&values, result.as_ref().ok(), &memo),
            case["state"],
            "case {number}"
        );
        // Break own fixture cycles after observing results; not engine GC.
        for value in &values {
            match value {
                _ if GeneralMapping::get(value).is_some() => value.clear_dict().unwrap(),
                Options::List(list) => list.borrow_mut().clear(),
                Options::Dict(_) => value.clear_dict().unwrap(),
                _ => (),
            }
        }
    }
}
#[test]
fn original_list_dict_copy_preserves_postorder_memo_aliases_opaque_tuples_key_substitution_and_partial_errors()
 {
    run(false, false);
    run(false, true);
}
#[test]
fn source_cyclic_copy_is_not_repaired_and_required_context_denial_preserves_completed_memo_and_balances_frames()
 {
    // Source recursion failures are retained separately. This verifies the
    // helper's required context-denial contract and completed memo, not that
    // the test budget equals the original process's remaining Python budget.
    run(true, false);
    run(true, true);
}
#[test]
fn original_general_dictionary_copy_preserves_none_tuple_object_keys_aliases_postorder_memo_substitution_and_partial_failures()
 {
    run_fixture(
        include_str!("../../fixtures/shared-general-deepcopy.json"),
        false,
        true,
        333,
    );
}
#[test]
fn general_dictionary_copy_context_denial_keeps_completed_memo_and_does_not_repair_source_cycles() {
    run_fixture(
        include_str!("../../fixtures/shared-general-deepcopy.json"),
        true,
        true,
        83,
    );
}
