use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    conditioning_preprocess as p,
    execution_value::Value as V,
    native::Error,
    patching::{
        hooks::{Hook, HookGroup, Kind},
        options::Options,
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::Device;

fn records(v: &Value) -> Vec<Options> {
    let V::List(values) = restore(v, Device::Cpu) else {
        panic!("Expected own record list")
    };
    values.into_iter().map(Options::from_value).collect()
}
fn snapshot(records: &[Options]) -> V {
    V::List(records.iter().map(|v| v.snapshot().unwrap()).collect())
}
fn aliases(records: &[Options], original: &[Options]) -> Value {
    let children: [(String, Value); 3] = ["model_conds", "mask", "shared"].map(|key| {
        (
            key.to_owned(),
            json!(
                records
                    .iter()
                    .map(|r| original
                        .iter()
                        .map(|o| {
                            match (r.get(key).unwrap(), o.get(key).unwrap()) {
                                (Some(a), Some(b)) => a.same_object(&b),
                                _ => false,
                            }
                        })
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
        )
    });
    json!({"records":records.iter().map(|r|original.iter().map(|o|r.same_object(o)).collect::<Vec<_>>()).collect::<Vec<_>>(),"children":children.into_iter().collect::<serde_json::Map<_,_>>()})
}
fn compare(result: Result<V, Error>, case: &Value, dev: Device, note: &str) {
    if case["error"].is_string() {
        reference_error(
            result
                .err()
                .unwrap_or_else(|| panic!("Expected error: {note}")),
            case,
        )
    } else {
        check(
            &result.unwrap_or_else(|e| panic!("{note}: {e}")),
            &case["output"],
            dev,
            note,
        );
    }
}
fn post(
    result: Result<(), Error>,
    values: &[Options],
    original: &[Options],
    case: &Value,
    dev: Device,
    note: &str,
) {
    compare(result.map(|_| V::None), case, dev, note);
    check(&snapshot(values), &case["post"], dev, note);
    assert_eq!(aliases(values, original), case["aliases"], "{note}");
}
#[test]
fn conditioning_preprocessing_preserves_masks_areas_timing_equal_area_fills_and_encoder_aliases() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-cond-preprocess.json")).unwrap();
    assert_eq!(fixture["boxes"].as_array().unwrap().len(), 72);
    assert_eq!(fixture["resolve"].as_array().unwrap().len(), 676);
    for (index, case) in fixture["boxes"].as_array().unwrap().iter().enumerate() {
        let input = restore(&case["input"], Device::Cpu).as_tensor().unwrap();
        compare(
            p::get_mask_aabb(&input),
            case,
            input.device(),
            &format!("boxes {index}"),
        );
    }
    for (index, case) in fixture["resolve"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let mut values = records(&case["initial"]);
        let original = values.clone();
        let dims = case["dims"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let dimensions = match case["dims_kind"].as_str().unwrap_or("list") {
            "size" => p::Dimensions::Size(&dims),
            "tuple" => p::Dimensions::Tuple(&dims),
            "list" => p::Dimensions::List(&dims),
            _ => unreachable!(),
        };
        let result = p::resolve_areas_and_cond_masks_multidim(&mut values, dimensions, dev);
        post(
            result,
            &values,
            &original,
            case,
            dev,
            &format!("resolve {index}"),
        );
    }
    for (index, case) in fixture["opposites"].as_array().unwrap().iter().enumerate() {
        let mut values = records(&case["initial"]);
        let source = Options::from_value(restore(&case["source"], Device::Cpu));
        let mut original = values.clone();
        original.push(source.clone());
        let result = p::create_cond_with_same_area_if_none(&mut values, &source);
        post(
            result,
            &values,
            &original,
            case,
            Device::Cpu,
            &format!("opposite {index}"),
        );
    }
    for (index, case) in fixture["timesteps"].as_array().unwrap().iter().enumerate() {
        let mut values = records(&case["initial"]);
        let original = values.clone();
        let mut calls = 0;
        let result = p::calculate_start_end_timesteps(&mut values, &mut |v| {
            let expected = &case["events"][calls];
            calls += 1;
            let value = v.snapshot()?;
            check(&value, expected, Device::Cpu, "percent");
            let number = match value {
                V::Float(x) => x,
                V::Int(x) => x as f64,
                _ => panic!("Numeric percent"),
            };
            Ok(Options::Atom(if number == 2. {
                V::None
            } else {
                V::Float(1. - number)
            }))
        });
        assert_eq!(calls, case["events"].as_array().unwrap().len());
        post(
            result,
            &values,
            &original,
            case,
            Device::Cpu,
            &format!("timesteps {index}"),
        );
    }
    for (index, case) in fixture["empty"].as_array().unwrap().iter().enumerate() {
        let source = records(&case["source"]);
        let mut values = records(&case["initial"]);
        let original = values.clone();
        let mut calls = 0;
        let result = p::apply_empty_x_to_equal_area(
            &source,
            &mut values,
            case["name"].as_str().unwrap(),
            &mut |cnets, i| {
                let expected = &case["events"][calls];
                calls += 1;
                assert_eq!(json!(i), expected["index"]);
                check(
                    &snapshot(cnets),
                    &expected["values"],
                    Device::Cpu,
                    "fill values",
                );
                Ok(cnets[i].clone())
            },
        );
        assert_eq!(calls, case["events"].as_array().unwrap().len());
        post(
            result,
            &values,
            &original,
            case,
            Device::Cpu,
            &format!("empty {index}"),
        );
    }
    for (index, case) in fixture["encodings"].as_array().unwrap().iter().enumerate() {
        let mut values = records(&case["initial"]);
        let original = values.clone();
        let dev = device(case["device"].as_str().unwrap());
        let noise = restore(&case["noise"], dev).as_tensor().unwrap();
        let kwargs = Options::from_value(restore(&case["kwargs"], dev));
        let mut calls = 0;
        let result = p::encode_model_conds(
            &mut values,
            &noise,
            dev,
            "positive",
            &kwargs,
            &mut |params| {
                let expected = &case["events"][calls];
                let i = calls;
                calls += 1;
                check(
                    &params.snapshot()?,
                    &expected["params"],
                    dev,
                    "encoder params",
                );
                assert_eq!(
                    params
                        .get("shared")?
                        .unwrap()
                        .same_object(&original[i].get("shared")?.unwrap()),
                    expected["shared_same"].as_bool().unwrap()
                );
                assert!(expected["noise_same"].as_bool().unwrap());
                let out = Options::from_value(V::Dict(vec![(
                    "base".into(),
                    V::List(vec![V::Text("replaced".into())]),
                )]));
                out.set("added", params.get("noise")?.unwrap())?;
                Ok(out)
            },
        );
        assert_eq!(calls, case["events"].as_array().unwrap().len());
        post(
            result,
            &values,
            &original,
            case,
            dev,
            &format!("encoding {index}"),
        );
    }
}
#[test]
fn registered_conditioning_hooks_preserve_group_identity_clones_and_none_counting() {
    let _native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-cond-preprocess.json")).unwrap();
    for case in fixture["filters"].as_array().unwrap() {
        let objects = [Kind::Weight, Kind::Weight, Kind::TransformerOptions]
            .map(|kind| Rc::new(RefCell::new(Hook::new(kind))));
        let groups = [vec![], vec![0], vec![0, 1], vec![1, 2], vec![0, 1, 2]].map(|ids| {
            let mut group = HookGroup::default();
            for i in ids {
                group.add(objects[i].clone());
            }
            Rc::new(RefCell::new(group))
        });
        let registered = case["registered"].as_array().map(|ids| {
            let mut group = HookGroup::default();
            for i in ids {
                group.add(Rc::new(RefCell::new(
                    objects[i.as_u64().unwrap() as usize]
                        .borrow()
                        .clone_source(),
                )));
            }
            Rc::new(RefCell::new(group))
        });
        let records: Vec<Vec<_>> = (0..2)
            .map(|branch| {
                let mut records = groups
                    .iter()
                    .map(|g| {
                        let out = Options::from_value(V::Dict(vec![]));
                        out.set("hooks", p::hook_option(Some(g.clone()))).unwrap();
                        out
                    })
                    .collect::<Vec<_>>();
                let out = Options::from_value(V::Dict(vec![]));
                if branch == 1 {
                    out.set("hooks", p::hook_option(None)).unwrap();
                }
                records.push(out);
                records
            })
            .collect();
        let original = records
            .iter()
            .flatten()
            .map(|r| r.get("hooks").unwrap().unwrap_or(p::hook_option(None)))
            .collect::<Vec<_>>();
        assert_eq!(
            json!(p::get_total_hook_groups_in_conds(&records).unwrap()),
            case["before"]
        );
        p::filter_registered_hooks_on_conds(&records, registered.as_ref()).unwrap();
        assert_eq!(
            json!(p::get_total_hook_groups_in_conds(&records).unwrap()),
            case["total"]
        );
        let after = records
            .iter()
            .flatten()
            .map(|r| r.get("hooks").unwrap().unwrap_or(p::hook_option(None)))
            .collect::<Vec<_>>();
        assert_eq!(
            json!(
                records
                    .iter()
                    .flatten()
                    .map(|r| r.get("hooks").unwrap().is_some())
                    .collect::<Vec<_>>()
            ),
            case["present"]
        );
        let same = |a: &Options, b: &Options| {
            matches!((a, b), (Options::Atom(V::None), Options::Atom(V::None))) || a.same_object(b)
        };
        assert_eq!(
            json!(
                after
                    .iter()
                    .map(|a| original.iter().map(|b| same(a, b)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            case["aliases"]
        );
        assert_eq!(
            json!(
                after
                    .iter()
                    .map(|a| after.iter().map(|b| same(a, b)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            case["sharing"]
        );
        let mut members = Vec::new();
        let mut exact = Vec::new();
        for v in after {
            if let Options::Object(v) = v {
                let g = v.downcast_ref::<RefCell<HookGroup>>().unwrap().borrow();
                members.push(Some(
                    g.hooks
                        .iter()
                        .map(|h| {
                            objects
                                .iter()
                                .position(|o| {
                                    Rc::ptr_eq(&h.borrow().reference, &o.borrow().reference)
                                })
                                .unwrap()
                        })
                        .collect::<Vec<_>>(),
                ));
                exact.push(Some(
                    g.hooks
                        .iter()
                        .map(|h| objects.iter().any(|o| Rc::ptr_eq(h, o)))
                        .collect::<Vec<_>>(),
                ));
            } else {
                members.push(None);
                exact.push(None);
            }
        }
        assert_eq!(json!(members), case["groups"]);
        assert_eq!(json!(exact), case["hook_same"]);
    }
}

struct Control {
    index: usize,
    extra: Vec<Option<imagine_inference::conditioning_batch::Hooks>>,
    clones: Vec<(Device, Rc<dyn p::Control>)>,
    events: Rc<RefCell<Vec<Value>>>,
}
impl p::Control for Control {
    fn extra_hooks(
        &self,
    ) -> Result<Vec<Option<imagine_inference::conditioning_batch::Hooks>>, Error> {
        self.events.borrow_mut().push(json!(["extra", self.index]));
        Ok(self.extra.clone())
    }
    fn pre_run(
        &self,
        model: &Options,
        percent: &mut dyn FnMut(&Options) -> Result<Options, Error>,
    ) -> Result<(), Error> {
        let V::Text(tag) = model.snapshot()? else {
            panic!("Own model tag")
        };
        self.events
            .borrow_mut()
            .push(json!(["pre", self.index, tag]));
        for p in [0.25, 0.75] {
            let out = percent(&Options::Atom(V::Float(p)))?;
            let V::Float(out) = out.snapshot()? else {
                panic!("Mapped percent")
            };
            self.events
                .borrow_mut()
                .push(json!(["mapped", self.index, p, out]));
        }
        Ok(())
    }
    fn multigpu_clones(&self) -> Result<Vec<(Device, Rc<dyn p::Control>)>, Error> {
        Ok(self.clones.clone())
    }
}
struct ControlModel {
    patcher: bool,
    events: Rc<RefCell<Vec<Value>>>,
}
impl p::ControlModel for ControlModel {
    fn model_object(&self) -> Options {
        Options::Atom(V::Text("main".into()))
    }
    fn multigpu_models(&mut self) -> Result<Vec<(Device, Options)>, Error> {
        if !self.patcher {
            return Ok(vec![]);
        }
        self.events
            .borrow_mut()
            .push(json!(["additional", "multigpu"]));
        Ok(vec![
            (Device::Cuda(0), Options::Atom(V::Text("old".into()))),
            (Device::Cuda(0), Options::Atom(V::Text("matched".into()))),
        ])
    }
    fn percent_to_sigma(&mut self, percent: &Options) -> Result<Options, Error> {
        let V::Float(p) = percent.snapshot()? else {
            panic!("Float percent")
        };
        self.events.borrow_mut().push(json!(["percent", p]));
        Ok(Options::Atom(V::Float(1. - p)))
    }
}
fn same_hooks(a: &Options, b: &Options) -> bool {
    matches!((a, b), (Options::Atom(V::None), Options::Atom(V::None))) || a.same_object(b)
}
#[test]
fn conditioning_control_extra_hooks_and_multigpu_prerun_order_match_original() {
    let _native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-cond-preprocess.json")).unwrap();
    assert_eq!(fixture["control_hooks"].as_array().unwrap().len(), 4);
    assert_eq!(fixture["pre_runs"].as_array().unwrap().len(), 8);
    for case in fixture["control_hooks"].as_array().unwrap() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let objects = [Kind::Weight, Kind::Weight, Kind::TransformerOptions]
            .map(|kind| Rc::new(RefCell::new(Hook::new(kind))));
        let groups = [vec![0], vec![1], vec![1, 2]].map(|ids| {
            let mut group = HookGroup::default();
            for i in ids {
                group.add(objects[i].clone());
            }
            Rc::new(RefCell::new(group))
        });
        let controls = (0..2)
            .map(|index| {
                let extra = match case["mode"].as_str().unwrap() {
                    "none" => vec![],
                    "one" => vec![Some(groups[index].clone())],
                    "multi" => vec![Some(groups[index].clone()), Some(groups[2].clone())],
                    "null_only" => vec![None],
                    _ => unreachable!(),
                };
                p::control_option(Rc::new(Control {
                    index,
                    extra,
                    clones: vec![],
                    events: events.clone(),
                }))
            })
            .collect::<Vec<_>>();
        let records = [
            (Some(0), Some(0)),
            (Some(0), Some(0)),
            (Some(0), None),
            (Some(1), Some(0)),
            (Some(1), None),
            (None, Some(0)),
            (None, None),
        ]
        .map(|(c, h)| {
            let out = Options::from_value(V::Dict(vec![]));
            if let Some(c) = c {
                out.set("control", controls[c].clone()).unwrap();
            }
            if let Some(h) = h {
                out.set("hooks", p::hook_option(Some(groups[h].clone())))
                    .unwrap();
            }
            out
        });
        let mut original = records
            .iter()
            .map(|r| r.get("hooks").unwrap().unwrap_or(p::hook_option(None)))
            .collect::<Vec<_>>();
        original.extend(groups.iter().map(|g| p::hook_option(Some(g.clone()))));
        let result = p::preprocess_conds_hooks(&[records[..4].to_vec(), records[4..].to_vec()]);
        compare(
            result.map(|_| V::None),
            case,
            Device::Cpu,
            "control hook combine",
        );
        assert_eq!(json!(*events.borrow()), case["events"]);
        assert_eq!(
            json!(
                records
                    .iter()
                    .map(|r| r.get("hooks").unwrap().is_some())
                    .collect::<Vec<_>>()
            ),
            case["present"]
        );
        let after = records
            .iter()
            .map(|r| r.get("hooks").unwrap().unwrap_or(p::hook_option(None)))
            .collect::<Vec<_>>();
        assert_eq!(
            json!(
                after
                    .iter()
                    .map(|a| original
                        .iter()
                        .map(|b| same_hooks(a, b))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            case["aliases"]
        );
        assert_eq!(
            json!(
                after
                    .iter()
                    .map(|a| after.iter().map(|b| same_hooks(a, b)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            case["sharing"]
        );
        let members = after
            .iter()
            .map(|v| {
                if let Options::Object(v) = v {
                    Some(
                        v.downcast_ref::<RefCell<HookGroup>>()
                            .unwrap()
                            .borrow()
                            .hooks
                            .iter()
                            .map(|h| {
                                objects
                                    .iter()
                                    .position(|o| {
                                        Rc::ptr_eq(&h.borrow().reference, &o.borrow().reference)
                                    })
                                    .unwrap()
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(members), case["groups"]);
    }
    for case in fixture["pre_runs"].as_array().unwrap() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let clones = (2..4)
            .map(|index| {
                Rc::new(Control {
                    index,
                    extra: vec![],
                    clones: vec![],
                    events: events.clone(),
                }) as Rc<dyn p::Control>
            })
            .collect::<Vec<_>>();
        let controls = (0..2)
            .map(|index| {
                let clones = if index == 0 {
                    vec![
                        (Device::Cuda(0), clones[0].clone()),
                        (Device::Cpu, clones[1].clone()),
                    ]
                } else {
                    vec![]
                };
                p::control_option(Rc::new(Control {
                    index,
                    extra: vec![],
                    clones,
                    events: events.clone(),
                }))
            })
            .collect::<Vec<_>>();
        let records = case["assignments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let out = Options::from_value(V::Dict(vec![]));
                if let Some(i) = v.as_u64() {
                    out.set("control", controls[i as usize].clone()).unwrap();
                }
                out
            })
            .collect::<Vec<_>>();
        let mut model = ControlModel {
            patcher: case["patcher"].as_bool().unwrap(),
            events: events.clone(),
        };
        compare(
            p::pre_run_control(&mut model, &records).map(|_| V::None),
            case,
            Device::Cpu,
            "control pre run",
        );
        assert_eq!(json!(*events.borrow()), case["events"]);
    }
}

struct ProcessLog {
    case: Value,
    index: Cell<usize>,
}
impl ProcessLog {
    fn next(&self, event: &str) -> &Value {
        let index = self.index.get();
        self.index.set(index + 1);
        let expected = &self.case["events"][index];
        assert_eq!(
            expected["event"], event,
            "{} event {index}",
            self.case["mode"]
        );
        expected
    }
}
struct ProcessControl {
    index: usize,
    clones: p::ControlClones,
    log: Rc<ProcessLog>,
}
impl p::Control for ProcessControl {
    fn extra_hooks(
        &self,
    ) -> Result<Vec<Option<imagine_inference::conditioning_batch::Hooks>>, Error> {
        panic!("process_conds does not implicitly preprocess control hooks")
    }
    fn pre_run(
        &self,
        model: &Options,
        percent: &mut dyn FnMut(&Options) -> Result<Options, Error>,
    ) -> Result<(), Error> {
        let expected = self.log.next("pre");
        assert_eq!(json!(self.index), expected["index"]);
        let V::Text(tag) = model.snapshot()? else {
            panic!("Own model tag")
        };
        assert_eq!(tag, expected["model"].as_str().unwrap());
        for p in [0.25, 0.75] {
            let out = percent(&Options::Atom(V::Float(p)))?;
            let expected = self.log.next("mapped");
            assert_eq!(json!(self.index), expected["index"]);
            assert_eq!(json!(p), expected["percent"]);
            let V::Float(sigma) = out.snapshot()? else {
                panic!("Own mapped sigma")
            };
            assert_eq!(json!(sigma), expected["sigma"]);
        }
        Ok(())
    }
    fn multigpu_clones(&self) -> Result<p::ControlClones, Error> {
        Ok(self.clones.clone())
    }
}
fn clean(value: &Options, controls: &[Options], hooks: &[Options]) -> V {
    match value {
        Options::Atom(value) => value.clone(),
        Options::List(values) => V::List(
            values
                .borrow()
                .iter()
                .map(|v| clean(v, controls, hooks))
                .collect(),
        ),
        Options::Tuple(values) => {
            V::Tuple(values.iter().map(|v| clean(v, controls, hooks)).collect())
        }
        Options::Dict(values) => V::Dict(
            values
                .borrow()
                .iter()
                .map(|(k, v)| (k.clone(), clean(v, controls, hooks)))
                .collect(),
        ),
        Options::Object(_) => {
            let (name, index) = if let Some(i) = controls.iter().position(|c| c.same_object(value))
            {
                ("control", i)
            } else if let Some(i) = hooks.iter().position(|h| h.same_object(value)) {
                ("hooks", i)
            } else {
                panic!("Unbound own fixture object")
            };
            V::Tuple(vec![V::Text(name.into()), V::Int(index as i64)])
        }
    }
}
struct ProcessModel {
    log: Rc<ProcessLog>,
    controls: Vec<Options>,
    hooks: Vec<Options>,
    dev: Device,
}
impl p::ControlModel for ProcessModel {
    fn model_object(&self) -> Options {
        Options::Atom(V::Text("main".into()))
    }
    fn multigpu_models(&mut self) -> Result<Vec<(Device, Options)>, Error> {
        let expected = self.log.next("additional");
        assert_eq!(expected["key"], "multigpu");
        Ok(vec![(
            Device::Cuda(0),
            Options::Atom(V::Text("matched".into())),
        )])
    }
    fn percent_to_sigma(&mut self, p: &Options) -> Result<Options, Error> {
        let expected = self.log.next("percent");
        let value = p.snapshot()?;
        check(&value, &expected["value"], Device::Cpu, "process percent");
        let number = match value {
            V::Float(x) => x,
            V::Int(x) => x as f64,
            _ => panic!("Own percent"),
        };
        Ok(Options::Atom(V::Float(1. - number)))
    }
}
impl p::ConditioningModel for ProcessModel {
    fn has_extra_conds(&self) -> bool {
        self.log.case["extra"].as_bool().unwrap()
    }
    fn extra_conds(&mut self, params: &Options) -> Result<Options, Error> {
        let expected = self.log.next("extra");
        check(
            &clean(params, &self.controls, &self.hooks),
            &expected["params"],
            self.dev,
            "process encoder params",
        );
        if self.log.case["mode"] == "encoder_error"
            && matches!(params.get("label")?.unwrap().snapshot()?,V::Text(s)if s=="p1")
        {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own synthetic encoder boundary".into(),
            });
        }
        Ok(Options::from_value(V::Dict(vec![
            ("base".into(), V::List(vec![V::Text("replaced".into())])),
            (
                "encoded".into(),
                V::tensor(
                    params
                        .get("noise")?
                        .unwrap()
                        .snapshot()?
                        .as_tensor()?
                        .f_mul_scalar(0.125)?,
                ),
            ),
        ])))
    }
}
fn flatten(conditions: &Options) -> (Vec<Options>, Vec<Options>) {
    let Options::Dict(dict) = conditions else {
        panic!("Own condition dictionary")
    };
    let lists = dict
        .borrow()
        .iter()
        .map(|(_, list)| list.clone())
        .collect::<Vec<_>>();
    let records = lists
        .iter()
        .flat_map(|l| {
            let Options::List(list) = l else {
                panic!("Own condition list")
            };
            list.borrow().clone()
        })
        .collect();
    (lists, records)
}
#[test]
fn whole_conditioning_preprocessing_preserves_order_partial_errors_and_shared_objects() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-cond-preprocess.json")).unwrap();
    assert_eq!(fixture["processes"].as_array().unwrap().len(), 80);
    for case in fixture["processes"].as_array().unwrap() {
        let dev = device(case["device"].as_str().unwrap());
        let log = Rc::new(ProcessLog {
            case: case.clone(),
            index: Cell::new(0),
        });
        let clone_controls = (2..4)
            .map(|index| {
                Rc::new(ProcessControl {
                    index,
                    clones: vec![],
                    log: log.clone(),
                }) as Rc<dyn p::Control>
            })
            .collect::<Vec<_>>();
        let controls = (0..2)
            .map(|index| {
                p::control_option(Rc::new(ProcessControl {
                    index,
                    log: log.clone(),
                    clones: if index == 0 {
                        vec![
                            (Device::Cuda(0), clone_controls[0].clone()),
                            (Device::Cpu, clone_controls[1].clone()),
                        ]
                    } else {
                        vec![]
                    },
                }))
            })
            .collect::<Vec<_>>();
        let hooks = [Kind::Weight, Kind::TransformerOptions].map(|kind| {
            let hook = Hook::new(kind);
            for (strength, percent) in [(1., 0.), (0.5, 0.5)] {
                hook.keyframes.borrow_mut().add(Rc::new(RefCell::new(
                    imagine_inference::patching::hooks::Keyframe::new(strength, percent, 1),
                )));
            }
            let mut group = HookGroup::default();
            group.add(Rc::new(RefCell::new(hook)));
            p::hook_option(Some(Rc::new(RefCell::new(group))))
        });
        let conditions = Options::from_value(restore(&case["initial"], dev));
        for record in flatten(&conditions).1 {
            for (key, objects) in [
                ("control", controls.as_slice()),
                ("hooks", hooks.as_slice()),
            ] {
                if let Some(value) = record.get(key).unwrap()
                    && let V::Int(index) = value.snapshot().unwrap()
                {
                    record.set(key, objects[index as usize].clone()).unwrap();
                }
            }
        }
        let (old_lists, original) = flatten(&conditions);
        let noise = restore(&case["noise"], dev).as_tensor().unwrap();
        let mut model = ProcessModel {
            log: log.clone(),
            dev,
            controls: controls.clone(),
            hooks: hooks.to_vec(),
        };
        let args = p::Arguments {
            seed: Options::Atom(V::Int(123)),
            ..Default::default()
        };
        let result = p::process_conds(&mut model, &noise, &conditions, dev, &args);
        if case["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("Expected process error {}", case["mode"])),
                case,
            )
        } else {
            assert_eq!(
                result.unwrap().same_object(&conditions),
                case["output_same"].as_bool().unwrap()
            );
        }
        assert_eq!(
            log.index.get(),
            case["events"].as_array().unwrap().len(),
            "{}",
            case["mode"]
        );
        check(
            &clean(&conditions, &controls, &hooks),
            &case["post"],
            dev,
            &format!("process post {}", case["mode"]),
        );
        let (lists, records) = flatten(&conditions);
        assert_eq!(
            json!(
                lists
                    .iter()
                    .map(|l| old_lists
                        .iter()
                        .map(|o| l.same_object(o))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            case["list_aliases"]
        );
        assert_eq!(
            aliases(&records, &original),
            case["aliases"],
            "process {}",
            case["mode"]
        );
    }
}
