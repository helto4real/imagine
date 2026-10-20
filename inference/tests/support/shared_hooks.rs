use super::kitchen;
use imagine_inference::{
    execution_value::Value as V,
    patching::{
        hooks::{self as h, Keyframe, Keyframes},
        options::Options,
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::Tensor;
fn state(g: &Keyframes) -> Value {
    json!({"index":g.current_index,"used":g.used_steps,"current_strength":g.current_strength,"strength":g.strength(),"last_t":g.last_t,"reset_curr_t":g.reset_curr_t,"current_percent":g.current.as_ref().map(|frame|frame.borrow().start_percent)})
}
#[test]
fn source_hook_keyframes_repeats_reset_clone_and_container_aliases_match() {
    let _native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-hooks.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut g = Keyframes::default();
        for frame in case["frames"].as_array().unwrap() {
            g.add(Rc::new(RefCell::new(Keyframe::new(
                frame[0].as_f64().unwrap(),
                frame[1].as_f64().unwrap(),
                frame[2].as_i64().unwrap(),
            ))));
        }
        for event in case["events"].as_array().unwrap() {
            match event["event"].as_str().unwrap() {
                "initialize" => g
                    .initialize_timesteps(&mut |p| Ok(V::Float(1. - p)))
                    .unwrap(),
                "prepare" => {
                    let max = Tensor::from(case["max_sigma"].as_f64().unwrap() as f32);
                    let changed = g.prepare(event["time"].as_f64().unwrap(), &max).unwrap();
                    assert_eq!(changed, event["changed"].as_bool().unwrap(), "{case}");
                }
                "reset" => g.reset(),
                "clone" => {
                    assert_eq!(state(&g.clone_source()), event["state"]);
                    continue;
                }
                _ => unreachable!(),
            }
            assert_eq!(state(&g), event["state"], "{case}");
        }
        assert_eq!(
            g.has_guarantee_steps(),
            case["guarantees"].as_bool().unwrap()
        );
    }
    for case in fixture["clones"].as_array().unwrap() {
        let kind = match case["name"].as_str().unwrap() {
            "Hook" => h::Kind::Base,
            "WeightHook" => h::Kind::Weight,
            "ObjectPatchHook" => h::Kind::ObjectPatch,
            "AdditionalModelsHook" => h::Kind::AdditionalModels,
            "TransformerOptionsHook" => h::Kind::TransformerOptions,
            "InjectionsHook" => h::Kind::Injections,
            _ => unreachable!(),
        };
        let mut hook = h::Hook::new(kind);
        if !case["values"].is_null() {
            hook.values = Options::from_value(V::List(
                case["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|_| V::Dict(vec![("own".into(), V::Int(1))]))
                    .collect(),
            ));
        }
        let cloned = hook.clone_source();
        assert_eq!(hook.same_hook(&cloned), case["equal"].as_bool().unwrap());
        assert_eq!(
            Rc::ptr_eq(&hook.reference, &cloned.reference),
            case["reference_same"].as_bool().unwrap()
        );
        assert_eq!(
            Rc::ptr_eq(&hook.keyframes, &cloned.keyframes),
            case["keyframes_same"].as_bool().unwrap()
        );
        if let Some(same) = case["values_same"].as_bool() {
            // None has shared singleton identity, while Atom is intentionally
            // value-owned; the list ownership boundary is what can mutate.
            if !case["values"].is_null() {
                assert_eq!(hook.values.same_object(&cloned.values), same);
            }
            if let Some(same) = case["child_same"].as_bool() {
                let (Options::List(a), Options::List(b)) = (&hook.values, &cloned.values) else {
                    panic!()
                };
                assert_eq!(a.borrow()[0].same_object(&b.borrow()[0]), same);
            }
        }
    }
}

#[test]
fn source_hook_groups_clip_ranges_cached_type_lists_and_combine_identity_match() {
    let _native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-hooks.json")).unwrap();
    for case in fixture["groups"].as_array().unwrap() {
        let mut group = h::HookGroup::default();
        let objects = [
            h::Kind::Weight,
            h::Kind::TransformerOptions,
            h::Kind::Weight,
            h::Kind::AdditionalModels,
        ]
        .map(|kind| Rc::new(RefCell::new(h::Hook::new(kind))));
        for (i, hook) in objects.iter().enumerate() {
            group.add(hook.clone());
            if i == 0 || i == 2 {
                for f in case["frames"][i / 2].as_array().unwrap() {
                    hook.borrow()
                        .keyframes
                        .borrow_mut()
                        .add(Rc::new(RefCell::new(Keyframe::new(
                            f[0].as_f64().unwrap(),
                            f[1].as_f64().unwrap(),
                            f[2].as_i64().unwrap(),
                        ))));
                }
            }
        }
        group.add(Rc::new(RefCell::new(objects[0].borrow().clone_source())));
        assert_eq!(
            group.hooks.len(),
            case["duplicate_count"].as_u64().unwrap() as usize
        );
        let schedule=group.clip_schedule().iter().map(|r| json!({"range":[r.range.0,r.range.1],"hooks":r.hooks.iter().map(|(hook,frame)| {
            let i=objects.iter().position(|h|Rc::ptr_eq(h,hook)).unwrap();
            let f=frame.as_ref().map(|f| {let f=f.borrow();json!([f.strength,f.start_percent,f.guarantee_steps])});
            json!({"index":i,"frame":f,"frame_same":frame.as_ref().is_none_or(|f|hook.borrow().keyframes.borrow().keyframes.iter().any(|v|Rc::ptr_eq(v,f)))})
        }).collect::<Vec<_>>()})).collect::<Vec<_>>();
        assert_eq!(json!(schedule), case["schedule"], "{case}");
        let group = Rc::new(RefCell::new(group));
        let one = h::HookGroup::combine_all(&[None, Some(group.clone()), None], 0)
            .unwrap()
            .unwrap();
        assert_eq!(
            Rc::ptr_eq(&one, &group),
            case["one_same"].as_bool().unwrap()
        );
        let both = h::HookGroup::combine_all(&[Some(group.clone()), Some(group.clone())], 0)
            .unwrap()
            .unwrap();
        assert_eq!(
            Rc::ptr_eq(&both, &group),
            case["both_same"].as_bool().unwrap()
        );
        assert_eq!(
            both.borrow().hooks.len(),
            case["both_count"].as_u64().unwrap() as usize
        );
        for (i, hook) in both.borrow().hooks.iter().enumerate() {
            assert_eq!(
                Rc::ptr_eq(&hook.borrow().reference, &objects[i].borrow().reference),
                case["both_refs"][i].as_bool().unwrap()
            );
            assert_eq!(
                Rc::ptr_eq(hook, &objects[i]),
                case["both_clones"][i].as_bool().unwrap()
            );
        }
    }
    for c in fixture["caches"].as_array().unwrap() {
        let mut group = h::HookGroup::default();
        let hook = Rc::new(RefCell::new(h::Hook::new(h::Kind::Weight)));
        group.add(hook.clone());
        let cached = group.get_type(h::Kind::Weight);
        assert_eq!(
            Rc::ptr_eq(&cached, &group.get_type(h::Kind::Weight)),
            c["before"]["present_same"].as_bool().unwrap()
        );
        assert_eq!(
            Rc::ptr_eq(
                &group.get_type(h::Kind::Injections),
                &group.get_type(h::Kind::Injections)
            ),
            c["before"]["missing_same"].as_bool().unwrap()
        );
        if c["mutate"].as_bool().unwrap() {
            hook.borrow_mut().hook_type = h::Kind::AdditionalModels;
        }
        let result = group.remove(&hook.borrow());
        if c["error"].is_string() {
            super::binding_tests::reference_error(result.err().unwrap(), c);
        } else {
            result.unwrap();
        }
        assert_eq!(group.hooks.len(), c["remaining"].as_u64().unwrap() as usize);
        assert_eq!(
            cached.borrow().len(),
            c["cached_count"].as_u64().unwrap() as usize
        );
        assert_eq!(
            Rc::ptr_eq(&cached, &group.get_type(h::Kind::Weight)),
            c["cached_still_same"].as_bool().unwrap()
        );
    }
    for c in fixture["combines"].as_array().unwrap() {
        let groups = (0..c["count"].as_u64().unwrap())
            .map(|_| Some(Rc::new(RefCell::new(h::HookGroup::default()))))
            .collect::<Vec<_>>();
        let result = h::HookGroup::combine_all(&groups, c["require"].as_i64().unwrap());
        if c["error"].is_string() {
            super::binding_tests::reference_error(result.err().unwrap(), c);
        } else {
            assert_eq!(result.unwrap().is_none(), c["none"].as_bool().unwrap());
        }
    }
}

fn options_json(value: &Options) -> Value {
    match value {
        Options::Dict(values) => Value::Object(
            values
                .borrow()
                .iter()
                .map(|(k, v)| (k.clone(), options_json(v)))
                .collect(),
        ),
        Options::List(values) => Value::Array(values.borrow().iter().map(options_json).collect()),
        Options::Atom(v) => match v {
            V::None => Value::Null,
            V::Int(n) => json!(n),
            V::Float(n) => json!(n),
            V::Bool(b) => json!(b),
            V::Text(s) => json!(s),
            _ => panic!("unexpected synthetic atom"),
        },
        _ => panic!("unexpected synthetic container"),
    }
}
fn from_json(value: &Value) -> Options {
    Options::from_value(match value {
        Value::Null => V::None,
        Value::Bool(b) => V::Bool(*b),
        Value::String(s) => V::Text(s.clone()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                V::Int(i)
            } else {
                V::Float(n.as_f64().unwrap())
            }
        }
        Value::Array(values) => {
            return Options::List(Rc::new(RefCell::new(
                values.iter().map(from_json).collect(),
            )));
        }
        Value::Object(values) => {
            return Options::Dict(Rc::new(RefCell::new(
                values
                    .iter()
                    .map(|(k, v)| (k.clone(), from_json(v)))
                    .collect(),
            )));
        }
    })
}
#[test]
fn source_hook_registration_preserves_predicates_lora_target_order_merges_and_partial_errors() {
    let _native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-hooks.json")).unwrap();
    struct Model {
        events: Rc<RefCell<Vec<Value>>>,
    }
    impl h::Registration for Model {
        fn load_weight_patches(
            &mut self,
            weights: &Options,
            clip: bool,
        ) -> Result<Options, imagine_inference::native::Error> {
            let weights = options_json(weights);
            self.events.borrow_mut().extend([
                json!(["keys", clip]),
                json!(["load",weights,{"clip":clip},false]),
            ]);
            Ok(from_json(&json!({"prepared":weights,"clip":clip})))
        }
        fn add_hook_patches(
            &mut self,
            _hook: &h::HookHandle,
            patches: &Options,
            strength: f64,
        ) -> Result<(), imagine_inference::native::Error> {
            self.events
                .borrow_mut()
                .push(json!(["add", options_json(patches), strength, 0]));
            Ok(())
        }
    }
    for c in fixture["registrations"].as_array().unwrap() {
        let kind = match c["name"].as_str().unwrap() {
            "Hook" => h::Kind::Base,
            "WeightHook" => h::Kind::Weight,
            "ObjectPatchHook" => h::Kind::ObjectPatch,
            "AdditionalModelsHook" => h::Kind::AdditionalModels,
            "TransformerOptionsHook" => h::Kind::TransformerOptions,
            "InjectionsHook" => h::Kind::Injections,
            _ => unreachable!(),
        };
        let mut hook = h::Hook::new(kind);
        hook.strength_model = 0.7;
        hook.strength_clip = -0.3;
        hook.need_weight_init = c["initialize"].as_bool().unwrap();
        hook.scope = if c["scope"] == "all" {
            h::Scope::AllConditioning
        } else {
            h::Scope::HookedOnly
        };
        hook.values = if kind == h::Kind::TransformerOptions {
            if c["bad_values"].as_bool().unwrap() {
                from_json(&Value::Null)
            } else {
                from_json(&json!({"patches":{"own":[1,2]}}))
            }
        } else {
            from_json(&json!({"value":"model"}))
        };
        hook.weights_clip = from_json(&json!({"value":"clip"}));
        let events = Rc::new(RefCell::new(Vec::new()));
        let observed = events.clone();
        let allow = c["allow"].as_bool().unwrap();
        hook.should_register = Rc::new(move |_, _, _, _, registered| {
            observed
                .borrow_mut()
                .push(json!(["should", registered.hooks.len()]));
            Ok(allow)
        });
        let hook = Rc::new(RefCell::new(hook));
        let mut model = Model {
            events: events.clone(),
        };
        let mut registered = h::HookGroup::default();
        let target = from_json(&json!({}));
        match c["target"].as_str().unwrap() {
            "model" | "clip" => target
                .set(
                    "target",
                    Options::Object(Rc::new(if c["target"] == "clip" {
                        h::WeightTarget::Clip
                    } else {
                        h::WeightTarget::Model
                    })),
                )
                .unwrap(),
            "text_clip" => target.set("target", from_json(&json!("clip"))).unwrap(),
            "none" => {}
            _ => unreachable!(),
        }
        let model_options = from_json(&json!({}));
        let result = h::register_hook(&hook, &mut model, &model_options, &target, &mut registered);
        if c["error"].is_string() {
            super::binding_tests::reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("expected source error {c}")),
                c,
            );
        } else {
            assert_eq!(result.unwrap(), c["output"].as_bool().unwrap(), "{c}");
        }
        assert_eq!(json!(*events.borrow()), c["events"], "{c}");
        assert_eq!(
            registered.hooks.len(),
            c["registered"].as_u64().unwrap() as usize,
            "{c}"
        );
        assert_eq!(options_json(&model_options), c["model_options"], "{c}");
        assert_eq!(
            hook.borrow().skip_adding,
            c["skip_adding"].as_bool().unwrap(),
            "{c}"
        );
        if c["apply_error"].is_string() {
            let options = from_json(&json!({"initial":1}));
            let error = h::transformer_options_from_hooks(Some(&registered), false, Some(options))
                .err()
                .expect("Original hook application rejection");
            super::binding_tests::reference_error(
                error,
                &json!({"error":c["apply_error"],"message":c["apply_message"]}),
            );
        }
        if c.get("applied").is_some() {
            let options = from_json(&json!({"initial":1}));
            let output =
                h::transformer_options_from_hooks(Some(&registered), false, Some(options.clone()))
                    .unwrap();
            assert_eq!(options_json(&output), c["applied"], "{c}");
            assert_eq!(
                output.same_object(&options),
                c["applied_same"].as_bool().unwrap(),
                "{c}"
            );
        }
    }
}
