use imagine_inference::patching::extensions::{CallbackList, OrderedRegistry};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
#[test]
fn patcher_callback_and_wrapper_lists_preserve_live_getters_fresh_misses_remove_clone_and_flatten_owners()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-registry-lists.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 64);
    for case in fixture["cases"].as_array().unwrap() {
        let tokens = (0_u32..8).map(Rc::new).collect::<Vec<_>>();
        let mut roots =
            BTreeMap::from([("original".to_owned(), OrderedRegistry::<u32>::default())]);
        let mut handles = Vec::<(String, CallbackList<u32>)>::new();
        for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let root = step["root"].as_str().unwrap_or("original");
            let op = step["op"].as_str().unwrap();
            let key = step["key"].as_str();
            let kind = step["kind"].as_str().unwrap_or("");
            let token = step["token"].as_u64().unwrap_or(0) as usize;
            let handle = step["handle"].as_str().unwrap_or("");
            match op {
                "add" | "plain_add" => {
                    roots
                        .get_mut(root)
                        .unwrap()
                        .add(kind, key, tokens[token].clone())
                }
                "remove" => roots.get_mut(root).unwrap().remove(kind, key),
                "clone" => {
                    roots.insert("clone".into(), roots[root].clone());
                }
                "get" | "all" => {
                    let list = if op == "get" {
                        roots[root].list(kind, key)
                    } else {
                        Rc::new(RefCell::new(roots[root].all(kind)))
                    };
                    if let Some((_, value)) = handles.iter_mut().find(|(k, _)| k == handle) {
                        *value = list;
                    } else {
                        handles.push((handle.into(), list));
                    }
                }
                "append" => handles
                    .iter()
                    .find(|(k, _)| k == handle)
                    .unwrap()
                    .1
                    .borrow_mut()
                    .push(tokens[token].clone()),
                "pop" => {
                    let mut list = handles
                        .iter()
                        .find(|(k, _)| k == handle)
                        .unwrap()
                        .1
                        .borrow_mut();
                    if !list.is_empty() {
                        list.remove(0);
                    }
                }
                _ => panic!("own action {op}"),
            }
            let actual = json!({
                "roots":roots.iter().map(|(name,registry)| (name.clone(),json!(["first","second","missing"].into_iter().map(|kind|(kind,registry.all(kind).iter().map(|t|**t).collect::<Vec<_>>())).collect::<BTreeMap<_,_>>()))).collect::<BTreeMap<_,_>>(),
                "handles":handles.iter().map(|(key,list)|(key,list.borrow().iter().map(|t|**t).collect::<Vec<_>>())).collect::<BTreeMap<_,_>>(),
                "aliases":handles.iter().map(|(_,a)|handles.iter().map(|(_,b)|Rc::ptr_eq(a,b)).collect::<Vec<_>>()).collect::<Vec<_>>()
            });
            assert_eq!(
                actual, step["expected"],
                "{} {} step {index} {op}",
                case["category"], case["seed"]
            );
            // Copying registry/list containers must preserve callable identity.
            for registry in roots.values() {
                for kind in ["first", "second", "missing"] {
                    for token in registry.all(kind) {
                        assert!(Rc::ptr_eq(&token, &tokens[*token as usize]));
                    }
                }
            }
        }
    }
}
