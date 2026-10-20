use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::{GeneralMapping, Options, copy_nested_dicts},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};

fn decode(
    value: &Value,
    nodes: &[Value],
    cache: &RefCell<Vec<Option<Options>>>,
    keys: &Rc<OwnMappingKeys>,
) -> Options {
    let Some(index) = value.get("ref") else {
        return Options::from_value(match value {
            Value::Null => V::None,
            Value::Number(v) => V::Int(v.as_i64().unwrap()),
            Value::String(v) => V::Text(v.clone()),
            _ => panic!("own scalar"),
        });
    };
    let index = index.as_u64().unwrap() as usize;
    if let Some(value) = cache.borrow()[index].as_ref() {
        return value.clone();
    }
    let node = &nodes[index];
    let result = if let Some(entries) = node.get("dict") {
        let result = keys.empty();
        cache.borrow_mut()[index] = Some(result.clone());
        for pair in entries.as_array().unwrap() {
            result
                .set_item(
                    decode(&pair[0], nodes, cache, keys),
                    decode(&pair[1], nodes, cache, keys),
                )
                .unwrap();
        }
        result
    } else if let Some(values) = node.get("list") {
        let result = Options::List(Rc::new(RefCell::new(vec![])));
        cache.borrow_mut()[index] = Some(result.clone());
        let children = values
            .as_array()
            .unwrap()
            .iter()
            .map(|v| decode(v, nodes, cache, keys))
            .collect();
        let Options::List(items) = &result else {
            unreachable!()
        };
        *items.borrow_mut() = children;
        result
    } else {
        Options::Tuple(Rc::new(
            node["tuple"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| decode(v, nodes, cache, keys))
                .collect(),
        ))
    };
    cache.borrow_mut()[index] = Some(result.clone());
    result
}
fn graph(roots: &[Options]) -> Value {
    fn visit(value: &Options, seen: &mut Vec<Options>, nodes: &mut Vec<Value>) -> Value {
        if value.is_dictionary() || matches!(value, Options::List(_) | Options::Tuple(_)) {
            if let Some(index) = seen.iter().position(|v| v.same_object(value)) {
                return json!({"ref":index});
            }
            let index = nodes.len();
            seen.push(value.clone());
            nodes.push(Value::Null);
            let result = if let Some(mapping) = GeneralMapping::get(value) {
                json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([visit(k,seen,nodes),visit(v,seen,nodes)])).collect::<Vec<_>>()})
            } else if let Options::List(values) = value {
                json!({"list":values.borrow().iter().map(|v|visit(v,seen,nodes)).collect::<Vec<_>>()})
            } else if let Options::Tuple(values) = value {
                json!({"tuple":values.iter().map(|v|visit(v,seen,nodes)).collect::<Vec<_>>()})
            } else {
                panic!("own dictionary representation")
            };
            nodes[index] = result;
            return json!({"ref":index});
        }
        match value {
            Options::Atom(V::None) => Value::Null,
            Options::Atom(V::Int(v)) => json!(v),
            Options::Atom(V::Text(v)) => json!(v),
            _ => panic!("own scalar snapshot"),
        }
    }
    let mut seen = vec![];
    let mut nodes = vec![];
    let roots = roots
        .iter()
        .map(|v| visit(v, &mut seen, &mut nodes))
        .collect::<Vec<_>>();
    json!({"roots":roots,"nodes":nodes})
}
#[test]
fn original_nested_dictionary_copy_preserves_full_initial_copy_arbitrary_keys_shallow_list_and_opaque_tuple_aliases_and_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-nested-copy.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 25);
    for (index, case) in cases.iter().enumerate() {
        let nodes = case["initial"]["nodes"].as_array().unwrap();
        let cache = RefCell::new(vec![None; nodes.len()]);
        let input = decode(
            &case["initial"]["roots"][0],
            nodes,
            &cache,
            &OwnMappingKeys::seeded(),
        );
        assert_eq!(
            graph(std::slice::from_ref(&input)),
            case["initial"],
            "input {index}"
        );
        let (output, error) = match copy_nested_dicts(&input) {
            Ok(value) => (value, Value::Null),
            Err(Error::ReferenceException { class, message }) => {
                (Options::Atom(V::None), json!([class, message]))
            }
            Err(error) => panic!("native copy {index}: {error:?}"),
        };
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(graph(&[input, output]), case["result"], "aliases {index}");
    }
}
