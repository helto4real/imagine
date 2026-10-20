use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        model_patcher::PatcherState,
        model_registry::{Registry, RegistryBindings},
        options::{GeneralMapping, Options, SourceText},
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
struct Keys(Rc<OwnMappingKeys>);
impl RegistryBindings for Keys {
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self.0.empty())
    }
}
struct Token(usize);
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn decode(value: &Value) -> Options {
    match value {
        Value::Null => Options::Atom(V::None),
        Value::String(v) => SourceText::from_utf8(v),
        Value::Array(v) => list(v.iter().map(decode).collect()),
        Value::Object(v) => Options::Tuple(Rc::new(
            v["tuple"].as_array().unwrap().iter().map(decode).collect(),
        )),
        _ => panic!("own key"),
    }
}
fn snap(value: &Options) -> Value {
    if let Some(text) = SourceText::get(value) {
        return json!(text.utf8().unwrap());
    }
    if let Some(mapping) = GeneralMapping::get(value) {
        return json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([snap(k),snap(v)])).collect::<Vec<_>>()});
    }
    match value {
        Options::Atom(V::None) => Value::Null,
        Options::Atom(V::Int(v)) => json!(v),
        Options::List(v) => json!({"list":v.borrow().iter().map(snap).collect::<Vec<_>>()}),
        Options::Tuple(v) => json!({"tuple":v.iter().map(snap).collect::<Vec<_>>()}),
        Options::Object(v) => json!({"token":v.downcast_ref::<Token>().unwrap().0}),
        _ => panic!("own registry value"),
    }
}
fn registry<'a>(state: &'a PatcherState, category: &str) -> &'a Registry {
    if category == "callbacks" {
        &state.registries.callbacks
    } else {
        &state.registries.wrappers
    }
}
#[test]
fn original_field_registries_preserve_preconstruction_views_rebinding_raw_keys_live_lists_fresh_misses_order_and_partial_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-registry-fields.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 24);
    for case in fixture["cases"].as_array().unwrap() {
        // Views exist before the first source attribute assignment.
        let state = PatcherState::uninitialized("ModelPatcher");
        let bindings = Keys(OwnMappingKeys::seeded());
        state.fields.set("callbacks", bindings.0.empty());
        state.fields.set("wrappers", bindings.0.empty());
        let category = case["category"].as_str().unwrap();
        let mut handles = Vec::<(String, Options)>::new();
        let mut roots = vec![];
        let tokens = (0..8)
            .map(|i| Options::Object(Rc::new(Token(i))))
            .collect::<Vec<_>>();
        for (i, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let kind = decode(&step["kind"]);
            let key = decode(&step["key"]);
            let token = &tokens[step["token"].as_u64().unwrap() as usize];
            let handle = step["handle"].as_str().unwrap();
            let methods = registry(&state, category);
            let result: Result<Option<Options>, Error> = (|| {
                match step["op"].as_str().unwrap() {
                    "add" => methods.add_with_key(&kind, &key, token.clone(), &bindings)?,
                    "plain" => methods.add(&kind, token.clone(), &bindings)?,
                    "get" => return Ok(Some(methods.get(&kind, &key, &bindings)?)),
                    "all" => return Ok(Some(methods.all(&kind, &bindings)?)),
                    "remove" => methods.remove_with_key(&kind, &key, &bindings)?,
                    "append" => {
                        let Options::List(v) =
                            &handles.iter().find(|(k, _)| k == handle).unwrap().1
                        else {
                            panic!("own handle")
                        };
                        v.borrow_mut().push(token.clone());
                    }
                    "replace" => {
                        if let Ok(root) = state.fields.get(category) {
                            roots.push(root);
                        }
                        state.fields.set(category, bindings.0.empty());
                    }
                    "delete" => state.fields.delete(category)?,
                    "bad-root" => state.fields.set(category, Options::Atom(V::Int(7))),
                    "bad-list" => {
                        let group = bindings.0.empty();
                        group.set_item(key.clone(), Options::Atom(V::Int(7)))?;
                        state.fields.get(category)?.set_item(kind.clone(), group)?;
                    }
                    op => panic!("own action {op}"),
                }
                Ok(None)
            })();
            let error = match result {
                Ok(value) => {
                    if let Some(value) = value {
                        if let Some((_, old)) = handles.iter_mut().find(|(k, _)| k == handle) {
                            *old = value;
                        } else {
                            handles.push((handle.into(), value));
                        }
                    }
                    Value::Null
                }
                Err(Error::ReferenceException { class, message }) => json!([class, message]),
                Err(error) => panic!(
                    "unbound registry case {category} {} step {i}: {error:?}",
                    case["seed"]
                ),
            };
            assert_eq!(
                error, step["error"],
                "error {category} {} step {i}",
                case["seed"]
            );
            assert_eq!(
                json!({"dict":state.fields.entries().iter().map(|(k,v)|json!([k,snap(v)])).collect::<Vec<_>>()}),
                step["fields"],
                "fields {category} {} step {i}",
                case["seed"]
            );
            assert_eq!(
                json!(roots.iter().map(snap).collect::<Vec<_>>()),
                step["roots"]
            );
            assert_eq!(
                json!(
                    handles
                        .iter()
                        .map(|(k, v)| (k.clone(), snap(v)))
                        .collect::<serde_json::Map<_, _>>()
                ),
                step["handles"]
            );
            assert_eq!(
                json!(
                    handles
                        .iter()
                        .map(|(_, x)| handles
                            .iter()
                            .map(|(_, y)| x.same_object(y))
                            .collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                ),
                step["aliases"]
            );
        }
    }
}
