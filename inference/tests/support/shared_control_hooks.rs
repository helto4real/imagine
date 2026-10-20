use super::{binding_tests::reference_error, kitchen};
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        self, Diagnostics, Patch, PatchValue,
        hook_patches::HookPatches,
        hooks::{Hook, HookGroup, HookHandle, Kind},
        options::Options as O,
    },
    sampling::control_hooks::{
        ControlHookBindings, ControlHookLocals, Group, get_hooks_from_cond, identity_hash_address,
        object_identity_hash,
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
};
use tch::Tensor;
struct Control {
    index: usize,
    hash: i64,
    extra: RefCell<Option<Group>>,
    previous: RefCell<Option<O>>,
}
struct ControlHandle(Rc<Control>);
struct Bindings {
    mode: String,
    events: RefCell<Vec<Value>>,
    reads: RefCell<BTreeMap<(usize, &'static str), usize>>,
    conditions: O,
    hash_calls: Cell<usize>,
    recursion: Cell<usize>,
    final_order: RefCell<Vec<usize>>,
    capacity: Cell<usize>,
}
fn control(v: &O) -> &Control {
    let O::Object(v) = v else {
        panic!("own native control")
    };
    if let Some(owner) = v.downcast_ref::<Control>() {
        owner
    } else {
        &v.downcast_ref::<ControlHandle>().unwrap().0
    }
}
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
impl Bindings {
    fn read(&self, c: &O, name: &'static str) -> Result<bool, Error> {
        if matches!(c, O::Atom(V::None)) {
            return Err(error(
                "AttributeError",
                format!("'NoneType' object has no attribute '{name}'"),
            ));
        }
        let i = control(c).index;
        let n = {
            let mut reads = self.reads.borrow_mut();
            let n = reads.entry((i, name)).or_default();
            *n += 1;
            *n
        };
        self.events.borrow_mut().push(json!([i, name, n]));
        let failure = if name == "extra_hooks" {
            "fail-extra"
        } else {
            "fail-previous"
        };
        if i == 0 && self.mode == failure && n == 1 {
            return Err(error("ValueError", format!("own {name} failure")));
        }
        if i == 0 && self.mode == "replace-list" && name == "extra_hooks" && n == 1 {
            let O::List(v) = &self.conditions else {
                unreachable!()
            };
            v.borrow_mut().clear();
        }
        let missing = if name == "extra_hooks" {
            "extra-second-none"
        } else {
            "previous-second-none"
        };
        Ok(!(i == 0 && self.mode == missing && n == 2))
    }
}
impl ControlHookBindings for Bindings {
    fn same_control(&self, a: &O, b: &O) -> bool {
        std::ptr::eq(control(a), control(b))
    }
    fn control_hash(&self, v: &O) -> Result<i64, Error> {
        self.hash_calls.set(self.hash_calls.get() + 1);
        if matches!(v, O::List(_)) {
            return Err(error("TypeError", "unhashable type: 'list'"));
        }
        Ok(control(v).hash)
    }
    fn extra_hooks(&self, c: &O) -> Result<Option<Group>, Error> {
        if !self.read(c, "extra_hooks")? {
            return Ok(None);
        }
        Ok(control(c).extra.borrow().clone())
    }
    fn previous_controlnet(&self, c: &O) -> Result<Option<O>, Error> {
        if !self.read(c, "previous_controlnet")? {
            return Ok(None);
        }
        Ok(control(c).previous.borrow().clone())
    }
    fn enter_control_recursion(&self) -> Result<(), Error> {
        // No fixture exhausts recursion. Production binds the execution
        // context's source recursion budget, rather than choosing a new limit.
        self.recursion.set(self.recursion.get() + 1);
        Ok(())
    }
    fn leave_control_recursion(&self) {
        self.recursion.set(self.recursion.get() - 1);
    }
    fn source_locals(&self, l: &ControlHookLocals, phase: &str) -> Result<(), Error> {
        if phase == "return" {
            *self.final_order.borrow_mut() = l.control_order().map(|v| control(v).index).collect();
            self.capacity.set(l.set_capacity());
            assert!(l.recursive_controls.is_empty());
        }
        Ok(())
    }
}
fn group(hooks: &[HookHandle]) -> Group {
    let mut group = HookGroup::default();
    for h in hooks {
        group.add(h.clone());
    }
    Rc::new(RefCell::new(group))
}
fn list(v: Vec<O>) -> O {
    O::List(Rc::new(RefCell::new(v)))
}
fn state(full: &Group, hooks: &[HookHandle]) -> Value {
    json!(full.borrow().hooks.iter().map(|v| {
        let vref=v.borrow();
        json!({"ref":hooks.iter().position(|h|Rc::ptr_eq(&h.borrow().reference,&vref.reference)).unwrap(),
            "same":hooks.iter().map(|h|Rc::ptr_eq(h,v)).collect::<Vec<_>>(),
            "keyframes":hooks.iter().map(|h|Rc::ptr_eq(&h.borrow().keyframes,&vref.keyframes)).collect::<Vec<_>>()})
    }).collect::<Vec<_>>())
}
#[test]
fn original_control_identity_set_and_extra_hook_collection_preserve_hashes_probes_resize_chains_aliases_and_errors()
 {
    let data: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-control-hooks.json")).unwrap();
    assert_eq!(data["trained_model_weights"], 0);
    assert_eq!(data["fixture_tensors"], 0);
    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() > 180);
    let mut maximum = 0;
    for (case_id, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let count = p["count"].as_u64().unwrap() as usize;
        maximum = maximum.max(count);
        let mode = p["mode"].as_str().unwrap();
        let identity = p["pattern"] == "identity";
        let note = format!("{case_id} {p}");
        let hooks = (0..if identity { count + 2 } else { 2 })
            .map(|_| Rc::new(RefCell::new(Hook::new(Kind::Base))))
            .collect::<Vec<_>>();
        let full = group(&[hooks.last().unwrap().clone()]);
        let groups = (0..if identity { count } else { 0 })
            .map(|i| {
                let g = group(&if mode == "empty-groups" {
                    vec![]
                } else {
                    vec![
                        hooks[i].clone(),
                        Rc::new(RefCell::new(hooks[i].borrow().clone_source())),
                    ]
                });
                if mode == "overlap" {
                    g.borrow_mut().add(hooks[count].clone());
                }
                g
            })
            .collect::<Vec<_>>();
        let controls = (0..count)
            .map(|i| {
                let v = Rc::new(Control {
                    index: i,
                    hash: case["hashes"][i].as_i64().unwrap(),
                    extra: RefCell::new(if mode == "no-hooks" {
                        None
                    } else {
                        Some(groups[i].clone())
                    }),
                    previous: RefCell::new(None),
                });
                if identity {
                    assert_eq!(
                        identity_hash_address(case["addresses"][i].as_u64().unwrap() as usize),
                        v.hash,
                        "{note} original object hash"
                    );
                    assert_eq!(
                        object_identity_hash(&v),
                        identity_hash_address(Rc::as_ptr(&v) as usize)
                    );
                }
                O::Object(v)
            })
            .collect::<Vec<_>>();
        if ["chain", "shared-chain", "previous-second-none"].contains(&mode) {
            for i in 0..count.saturating_sub(1) {
                *control(&controls[i]).previous.borrow_mut() = Some(controls[i + 1].clone());
            }
        }
        if mode == "shared-chain" && count > 2 {
            *control(&controls[0]).previous.borrow_mut() = Some(controls[count - 1].clone());
        }
        if mode == "full-alias" && count > 0 {
            *control(&controls[0]).extra.borrow_mut() = Some(full.clone());
        }
        if mode == "shared-group" && count > 0 {
            for c in &controls {
                *control(c).extra.borrow_mut() = Some(groups[0].clone());
            }
        }
        let mut base = (0..count).collect::<Vec<_>>();
        if mode == "reverse" {
            base.reverse();
        }
        if mode == "duplicates" && count > 0 {
            base.extend([count - 1, 0]);
            base.extend(0..count);
        }
        let mut conditions = base
            .iter()
            .map(|&i| {
                let value = if mode == "duplicates" {
                    let O::Object(owner) = &controls[i] else {
                        unreachable!()
                    };
                    // Distinct native binding wrappers for the same canonical
                    // control must retain original set identity and dedup.
                    O::Object(Rc::new(ControlHandle(
                        owner
                            .clone()
                            .downcast::<Control>()
                            .unwrap_or_else(|_| panic!("own control")),
                    )))
                } else {
                    controls[i].clone()
                };
                O::dictionary(vec![("control".into(), value)])
            })
            .collect::<Vec<_>>();
        if identity {
            conditions.insert(
                0,
                O::dictionary(vec![(
                    "hooks".into(),
                    O::Object(if count > 0 {
                        groups[0].clone()
                    } else {
                        group(&[])
                    }),
                )]),
            );
            conditions.push(O::dictionary(vec![(
                "hooks".into(),
                O::Object(full.clone()),
            )]));
        }
        if mode == "bad-hooks" {
            conditions.insert(1, O::dictionary(vec![("hooks".into(), O::Atom(V::None))]));
        }
        if mode == "bad-control" {
            conditions.push(O::dictionary(vec![("control".into(), list(vec![]))]));
        }
        let conditions = if p["layout"] == "tuple" {
            O::Tuple(Rc::new(conditions))
        } else {
            list(conditions)
        };
        let bindings = Bindings {
            mode: mode.into(),
            events: RefCell::new(vec![]),
            reads: RefCell::new(BTreeMap::new()),
            conditions: conditions.clone(),
            hash_calls: Cell::new(0),
            recursion: Cell::new(0),
            final_order: RefCell::new(vec![]),
            capacity: Cell::new(0),
        };
        if identity {
            assert_eq!(state(&full, &hooks), case["before"], "{note} initial");
        }
        match get_hooks_from_cond(&conditions, full.clone(), &bindings) {
            Ok(v) => {
                assert!(case.get("error").is_none(), "{note} expected {case}");
                assert!(Rc::ptr_eq(&full, &v));
                assert_eq!(
                    json!(*bindings.final_order.borrow()),
                    case["order"],
                    "{note} exact captured set order"
                );
                assert_eq!(
                    json!(bindings.capacity.get()),
                    case["capacity"],
                    "{note} source table capacity"
                );
                assert_eq!(
                    bindings.hash_calls.get(),
                    base.len(),
                    "{note} duplicates are hashed again"
                );
            }
            Err(e) => {
                reference_error(e, case);
            }
        }
        assert_eq!(bindings.recursion.get(), 0, "{note} recursive exits");
        if identity {
            assert_eq!(
                json!(*bindings.events.borrow()),
                case["events"],
                "{note} repeated control attribute read order"
            );
            assert_eq!(
                state(&full, &hooks),
                case["after"],
                "{note} exact hook identity/keyframe sharing and partial results"
            );
        }
    }
    assert_eq!(
        maximum, 78644,
        "large-set growth factor boundary must remain exercised"
    );
}
#[test]
fn approved_control_hook_orders_keep_exact_combined_patch_bytes_for_both_reference_orders() {
    let diagnostic: Value = serde_json::from_str(include_str!(
        "../../../design/stage-3-evidence/control-hook-set-order-diagnostic.json"
    ))
    .unwrap();
    let source: Value = serde_json::from_str(include_str!(
        "../../../design/stage-3-evidence/control-hook-set-order-weight-probe.json"
    ))
    .unwrap();
    let native = kitchen();
    for case in source["results"].as_array().unwrap() {
        let hooks = (0..2)
            .map(|_| Rc::new(RefCell::new(Hook::new(Kind::Base))))
            .collect::<Vec<_>>();
        let mut patches = HookPatches::default();
        for (i, h) in hooks.iter().enumerate() {
            let diff = O::Atom(V::tensor(Tensor::from_slice(&[if i == 0 {
                16777216_f32
            } else {
                -16777216_f32
            }])));
            patches.set(
                h.borrow().reference.clone(),
                O::dictionary(vec![(
                    "w".into(),
                    list(vec![O::Tuple(Rc::new(vec![
                        O::Atom(V::Float(1.)),
                        O::Tuple(Rc::new(vec![diff])),
                        O::Atom(V::Float(1.)),
                        O::Atom(V::None),
                        O::Atom(V::None),
                    ]))]),
                )]),
            );
        }
        let captured = diagnostic["runs"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|run| run["cases"].as_array().unwrap())
            .find(|v| v["count"] == 2 && v["padding"] == 0 && v["order"] == case["order"])
            .expect("each numerical order was observed with original identity hashes");
        let controls = (0..2)
            .map(|i| {
                O::Object(Rc::new(Control {
                    index: i,
                    hash: captured["identity_hashes"][i].as_i64().unwrap(),
                    extra: RefCell::new(Some(group(&[hooks[i].clone()]))),
                    previous: RefCell::new(None),
                }))
            })
            .collect::<Vec<_>>();
        let conditions = list(
            [0, 1, 1, 0]
                .iter()
                .map(|&i| O::dictionary(vec![("control".into(), controls[i].clone())]))
                .collect(),
        );
        let bindings = Bindings {
            mode: "ordinary".into(),
            events: RefCell::new(vec![]),
            reads: RefCell::new(BTreeMap::new()),
            conditions: conditions.clone(),
            hash_calls: Cell::new(0),
            recursion: Cell::new(0),
            final_order: RefCell::new(vec![]),
            capacity: Cell::new(0),
        };
        let ordered = get_hooks_from_cond(&conditions, group(&[]), &bindings).unwrap();
        assert_eq!(json!(*bindings.final_order.borrow()), case["order"]);
        assert_eq!(ordered.borrow().hooks.len(), 2);
        let combined = patches.combined(Some(&ordered.borrow())).unwrap();
        let O::List(values) = combined.get("w").unwrap().unwrap() else {
            unreachable!()
        };
        let native_patches = values
            .borrow()
            .iter()
            .map(|p| {
                let O::Tuple(p) = p else { unreachable!() };
                let O::Tuple(diff) = &p[1] else {
                    unreachable!()
                };
                Patch::new(
                    PatchValue::Diff {
                        weight: diff[0].snapshot().unwrap(),
                        pad: false,
                    },
                    1.,
                )
            })
            .collect::<Vec<_>>();
        let output = patching::calculate_weight(
            &native,
            &native_patches,
            Tensor::from_slice(&[1_f32]),
            "synthetic.w",
            tch::Kind::Float,
            None,
            &mut Diagnostics::default(),
        )
        .unwrap();
        let value = output.double_value(&[0]) as f32;
        let hex = value
            .to_le_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            hex,
            case["payload_hex"].as_str().unwrap(),
            "exact bytes for matched original hook order"
        );
        assert_eq!(json!([value]), case["values"]);
    }
}
