use super::{
    binding_tests::{check, reference_error, restore},
    kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V, native::Error, patching::options::Options,
    sampling::preparation as p,
};
use serde_json::{Value, json};
use std::rc::Rc;
use tch::{Device, Kind};
fn compare(result: Result<V, Error>, expected: &Value, note: &str) {
    if expected["error"].is_string() {
        reference_error(
            result
                .err()
                .unwrap_or_else(|| panic!("Expected source error {note}")),
            expected,
        )
    } else {
        check(
            &result.unwrap_or_else(|e| panic!("{note}: {e}")),
            &expected["output"],
            Device::Cpu,
            note,
        )
    }
}
struct Estimate {
    case: Value,
    index: usize,
    previous: Option<Options>,
}
impl Estimate {
    fn next(&mut self, event: &str) -> Value {
        let expected = self.case["events"][self.index].clone();
        self.index += 1;
        assert_eq!(expected["event"], event);
        expected
    }
}
impl p::MemoryEstimator for Estimate {
    fn extra_conds_shapes(&mut self, record: &Options) -> Result<Options, Error> {
        let expected = self.next("extra");
        let V::Text(label) = record.get("label")?.unwrap().snapshot()? else {
            panic!("Own label")
        };
        assert_eq!(label, expected["label"].as_str().unwrap());
        if self.case["pattern"] == "mutating"
            && let Some(previous) = &self.previous
        {
            let Options::List(cross) = previous.get("cross")?.unwrap() else {
                panic!("Own mutable shape")
            };
            cross.borrow_mut()[1] = Options::Atom(V::Int(40));
        }
        let out = record.get("dims")?.unwrap();
        self.previous = Some(out.clone());
        Ok(out)
    }
    fn memory_required(&mut self, shape: &[i64], conditions: &Options) -> Result<f64, Error> {
        let expected = self.next("memory");
        assert_eq!(json!(shape), expected["shape"]);
        check(
            &conditions.snapshot()?,
            &expected["conditions"],
            Device::Cpu,
            "estimator shape arguments",
        );
        let V::Dict(values) = conditions.snapshot()? else {
            panic!("Own shape mapping")
        };
        let mut total = 0_i64;
        for (_, values) in values {
            let V::List(values) = values else {
                panic!("Own list")
            };
            for value in values {
                let shape = match value {
                    V::List(v) | V::Tuple(v) => v,
                    _ => panic!("Own shape"),
                };
                total += shape
                    .iter()
                    .map(|n| match n {
                        V::Int(n) => *n,
                        _ => panic!("Own dimension"),
                    })
                    .product::<i64>();
            }
        }
        Ok((shape.iter().product::<i64>() * 4) as f64 + total as f64 * 0.125)
    }
}
struct Access {
    events: Vec<Value>,
}
fn control_index(value: &Options) -> Result<usize, Error> {
    if let Options::Object(value) = value {
        return Ok(*value.downcast_ref::<usize>().unwrap());
    }
    Err(Error::ReferenceException {
        class: "AttributeError",
        message: "'NoneType' object has no attribute 'get_models'".into(),
    })
}
impl p::Dependencies for Access {
    fn control_models(&mut self, control: &Options) -> Result<Vec<Options>, Error> {
        let i = control_index(control)?;
        self.events.push(json!(["models", i]));
        Ok((0..2)
            .map(|_| Options::Atom(V::Text(format!("model-{i}"))))
            .collect())
    }
    fn inference_memory_requirements(
        &mut self,
        control: &Options,
        dtype: Kind,
    ) -> Result<f64, Error> {
        let i = control_index(control)?;
        let dtype = match dtype {
            Kind::Float => "torch.float32",
            Kind::BFloat16 => "torch.bfloat16",
            _ => unreachable!(),
        };
        self.events.push(json!(["memory", i, dtype]));
        Ok((i + 1) as f64 * 64.)
    }
    fn cleanup_if_present(&mut self, model: &Options) -> Result<(), Error> {
        let V::Int(index) = model.snapshot()? else {
            panic!("Own model index")
        };
        if index == 0 {
            self.events.push(json!(0));
        }
        Ok(())
    }
}
fn bind_controls(conditions: &Options, controls: &[Options]) {
    let Options::Dict(groups) = conditions else {
        panic!("Own groups")
    };
    for (_, group) in groups.borrow().iter() {
        let Options::List(rows) = group else {
            panic!("Own group list")
        };
        for record in rows.borrow().iter() {
            if let Some(value) = record.get("control").unwrap() {
                let replace = |v: &Options| match v.snapshot().unwrap() {
                    V::Int(i) => controls[i as usize].clone(),
                    _ => v.clone(),
                };
                let new = if let Options::List(values) = value {
                    Options::List(Rc::new(std::cell::RefCell::new(
                        values.borrow().iter().map(replace).collect(),
                    )))
                } else {
                    replace(&value)
                };
                record.set("control", new).unwrap();
            }
        }
    }
}
#[test]
fn sampling_preparation_preserves_shared_shape_maps_control_order_dependency_duplicates_and_cleanup()
 {
    let _native = kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-preparation.json")).unwrap();
    assert_eq!(fixture["estimates"].as_array().unwrap().len(), 24);
    assert_eq!(fixture["dependencies"].as_array().unwrap().len(), 16);
    for (i, case) in fixture["estimates"].as_array().unwrap().iter().enumerate() {
        let conditions = Options::from_value(restore(&case["initial"], Device::Cpu));
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap())
            .collect::<Vec<_>>();
        let mut model = Estimate {
            case: case.clone(),
            index: 0,
            previous: None,
        };
        let result = p::estimate_memory(
            &mut model,
            imagine_inference::conditioning_preprocess::Dimensions::List(&shape),
            &conditions,
        )
        .map(|(a, b)| V::Tuple(vec![V::Float(a), V::Float(b)]));
        compare(result, case, &format!("estimate {i}"));
        assert_eq!(model.index, case["events"].as_array().unwrap().len());
        check(
            &conditions.snapshot().unwrap(),
            &case["post"],
            Device::Cpu,
            "source input shape mutations",
        );
    }
    for (i, case) in fixture["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let conditions = Options::from_value(restore(&case["initial"], Device::Cpu));
        let controls = (0..2)
            .map(|i| Options::Object(Rc::new(i as usize)))
            .collect::<Vec<_>>();
        bind_controls(&conditions, &controls);
        let mut access = Access { events: vec![] };
        let result = p::get_additional_models(
            &conditions,
            kind(case["kind"].as_str().unwrap()),
            &mut access,
        )
        .and_then(|(models, memory)| {
            Ok(V::Tuple(vec![
                V::List(
                    models
                        .iter()
                        .map(Options::snapshot)
                        .collect::<Result<_, _>>()?,
                ),
                V::Float(memory),
            ]))
        });
        compare(result, case, &format!("dependencies {i}"));
        assert_eq!(json!(access.events), case["events"]);
    }
    for case in fixture["models_from"].as_array().unwrap() {
        let V::List(records) = restore(&case["initial"], Device::Cpu) else {
            panic!("Own list")
        };
        let records = records
            .into_iter()
            .map(Options::from_value)
            .collect::<Vec<_>>();
        compare(
            p::get_models_from_cond(&records, "control").and_then(|models| {
                Ok(V::List(
                    models
                        .iter()
                        .map(Options::snapshot)
                        .collect::<Result<_, _>>()?,
                ))
            }),
            case,
            "get_models_from_cond",
        );
    }
    for case in fixture["cleanups"].as_array().unwrap() {
        let models = case["identities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| Options::Atom(V::Int(v.as_i64().unwrap())))
            .collect::<Vec<_>>();
        let mut access = Access { events: vec![] };
        compare(
            p::cleanup_additional_models(&models, &mut access).map(|_| V::None),
            case,
            "cleanup",
        );
        assert_eq!(json!(access.events), case["events"]);
    }
    for case in fixture["hook_models"].as_array().unwrap() {
        use imagine_inference::patching::hooks::{Hook, HookGroup, Kind};
        let options = Options::from_value(V::Dict(vec![]));
        if case["mode"] == "null" {
            options
                .set("registered_hooks", Options::Atom(V::None))
                .unwrap();
        } else if case["mode"] == "group" {
            let mut group = HookGroup::default();
            group.add(Rc::new(std::cell::RefCell::new(Hook::new(Kind::Weight))));
            let V::List(values) = restore(&case["values"], Device::Cpu) else {
                panic!("Own hook models")
            };
            for value in values {
                let mut hook = Hook::new(Kind::AdditionalModels);
                hook.values = Options::from_value(value);
                group.add(Rc::new(std::cell::RefCell::new(hook)));
            }
            options
                .set(
                    "registered_hooks",
                    Options::Object(Rc::new(std::cell::RefCell::new(group))),
                )
                .unwrap();
        }
        let input = if case["mode"] == "none" {
            None
        } else {
            Some(&options)
        };
        compare(
            p::get_additional_models_from_model_options(input).and_then(|models| {
                Ok(V::List(
                    models
                        .iter()
                        .map(Options::snapshot)
                        .collect::<Result<_, _>>()?,
                ))
            }),
            case,
            "registered additional models",
        );
    }
}
