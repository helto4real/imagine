use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{
    conditioning::{AreaCondition, Condition, Kind as CKind},
    conditioning_batch::{self as b, Call, Model},
    execution_value::Value as V,
    native::Error,
    patching::{hooks::HookGroup, options::Options},
};
use serde_json::Value;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::{Device, Kind, Tensor};
fn data(shape: &[i64], dev: Device, dtype: Kind, offset: i64) -> Tensor {
    Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, dev))
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .reshape(shape)
        .to_kind(dtype)
}
struct Log {
    case: Value,
    index: Cell<usize>,
}
impl Log {
    fn next(&self, event: &str) -> &Value {
        let i = self.index.get();
        let v = &self.case["events"][i];
        assert_eq!(v["event"], event, "{} event {i}", self.case["mode"]);
        self.index.set(i + 1);
        v
    }
}
struct Runtime {
    log: Rc<Log>,
    hooks: Vec<b::Hooks>,
}
impl Runtime {
    fn hook_id(&self, h: Option<&b::Hooks>) -> Value {
        h.map_or(Value::Null, |h| {
            Value::from(
                self.hooks
                    .iter()
                    .position(|other| Rc::ptr_eq(h, other))
                    .unwrap(),
            )
        })
    }
}
impl Model for Runtime {
    fn prepare_hook_keyframe(
        &mut self,
        time: &Tensor,
        h: &b::Hooks,
        _opts: &Options,
    ) -> Result<(), Error> {
        let v = self.log.next("prepare_hook");
        assert_eq!(self.hook_id(Some(h)), v["hook"]);
        check(
            &V::tensor(time.shallow_clone()),
            &v["time"],
            time.device(),
            "hook time",
        );
        Ok(())
    }
    fn prepare_state(&mut self, time: &Tensor, _opts: &Options) -> Result<(), Error> {
        let v = self.log.next("prepare_state");
        check(
            &V::tensor(time.shallow_clone()),
            &v["time"],
            time.device(),
            "prepare time",
        );
        Ok(())
    }
    fn free_memory(&mut self, dev: Device) -> Result<f64, Error> {
        let v = self.log.next("free");
        assert_eq!(format_device(dev), v["device"].as_str().unwrap());
        Ok(self.log.case["free"].as_f64().unwrap())
    }
    fn memory_required(
        &mut self,
        shape: &[i64],
        conditions: &[(String, Vec<V>)],
    ) -> Result<f64, Error> {
        let v = self.log.next("estimate");
        assert_eq!(serde_json::json!(shape), v["shape"]);
        check(
            &V::Dict(
                conditions
                    .iter()
                    .map(|(k, v)| (k.clone(), V::List(v.clone())))
                    .collect(),
            ),
            &v["conditions"],
            Device::Cpu,
            "condition sizes",
        );
        Ok((shape[0] * shape[2] * shape[3] * 64) as f64)
    }
    fn apply_hooks(&mut self, h: Option<&b::Hooks>) -> Result<Options, Error> {
        let id = self.hook_id(h);
        let v = self.log.next("hooks");
        assert_eq!(id, v["hook"]);
        Ok(Options::from_value(V::Dict(vec![
            (
                "branch".into(),
                V::List(vec![id.as_i64().map_or(V::None, V::Int)]),
            ),
            (
                "patches".into(),
                V::Dict(vec![(
                    "base_patch".into(),
                    V::List(vec![V::Text("own-base".into())]),
                )]),
            ),
        ])))
    }
    fn apply_model(&mut self, call: Call<'_>, _opts: &Options) -> Result<Tensor, Error> {
        let v = self.log.next("model");
        let dev = call.input.device();
        check(
            &V::tensor(call.input.shallow_clone()),
            &v["input"],
            dev,
            "model input",
        );
        check(
            &V::tensor(call.timestep.shallow_clone()),
            &v["time"],
            dev,
            "model timestep",
        );
        check(
            &call.conditioning.snapshot()?,
            &v["conditioning"],
            dev,
            "model conditioning",
        );
        let cross = call
            .conditioning
            .get("c_crossattn")?
            .unwrap()
            .snapshot()?
            .as_tensor()?;
        let mut y = call
            .input
            .f_mul_scalar(0.125)?
            .f_add(&call.timestep.f_reshape([-1, 1, 1, 1])?.f_mul_scalar(0.01)?)?
            .f_add(&cross.f_mean(None)?.f_mul_scalar(0.03125)?)?;
        if let Some(control) = call.conditioning.get("control")? {
            let V::List(list) = control.get("input")?.unwrap().snapshot()? else {
                panic!()
            };
            y = y.f_add(&list[0].as_tensor()?.f_mul_scalar(0.0625)?)?;
        }
        Ok(y)
    }
    fn warning(&mut self, message: &str) {
        panic!("Unexpected original warning {message}")
    }
}
fn format_device(dev: Device) -> String {
    match dev {
        Device::Cpu => "cpu".into(),
        Device::Cuda(n) => format!("cuda:{n}"),
        _ => unreachable!(),
    }
}
struct Control {
    index: usize,
    log: Rc<Log>,
}
impl b::Control for Control {
    fn get_control(
        &self,
        x: &Tensor,
        t: &Tensor,
        c: &Options,
        chunks: usize,
        transformer: &Options,
    ) -> Result<Options, Error> {
        let v = self.log.next("control");
        assert_eq!(self.index as u64, v["index"].as_u64().unwrap());
        assert_eq!(chunks as u64, v["chunks"].as_u64().unwrap());
        for (key, tensor) in [("input", x), ("time", t)] {
            check(&V::tensor(tensor.shallow_clone()), &v[key], x.device(), key);
        }
        check(
            &c.snapshot()?,
            &v["conditioning"],
            x.device(),
            "control conditioning",
        );
        check(
            &transformer.snapshot()?,
            &v["transformer"],
            x.device(),
            "control transformer",
        );
        Ok(Options::from_value(V::Dict(vec![
            (
                "input".into(),
                V::List(vec![V::tensor(x.f_mul_scalar(0.25)?)]),
            ),
            ("middle".into(), V::List(vec![])),
            ("output".into(), V::List(vec![])),
        ])))
    }
}
struct Gligen {
    kind: String,
    log: Rc<Log>,
}
impl b::Gligen for Gligen {
    fn patch(&self, shape: &[i64], dev: Device) -> Result<Options, Error> {
        let v = self.log.next("gligen");
        assert_eq!(self.kind, v["kind"]);
        assert_eq!(serde_json::json!(shape), v["shape"]);
        assert_eq!(format_device(dev), v["device"].as_str().unwrap());
        Ok(Options::from_value(V::Text(format!("own-{}", self.kind))))
    }
}
#[test]
fn conditioning_batch_preserves_original_memory_fit_order_defaults_controls_hooks_and_gligen() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-cond-batch.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["kind"].as_str().unwrap());
        let log = Rc::new(Log {
            case: c.clone(),
            index: Cell::new(0),
        });
        let hooks = (0..2)
            .map(|_| Rc::new(RefCell::new(HookGroup::default())))
            .collect::<Vec<_>>();
        let controls = (0..2)
            .map(|index| {
                Rc::new(Control {
                    index,
                    log: log.clone(),
                }) as Rc<dyn b::Control>
            })
            .collect::<Vec<_>>();
        let mut groups = Vec::new();
        for descriptions in c["descriptions"].as_array().unwrap() {
            if descriptions.is_null() {
                groups.push(None);
                continue;
            }
            let mut rows = Vec::new();
            for d in descriptions.as_array().unwrap() {
                rows.push(b::Record {
                    area: AreaCondition {
                        area: d["area"]
                            .as_array()
                            .map(|a| a.iter().map(|n| n.as_i64().unwrap()).collect()),
                        strength: 0.75,
                        mask: if d["mask"].is_boolean() {
                            Some(data(&[1, 12, 16], dev, Kind::Float, 5).abs())
                        } else {
                            None
                        },
                        mask_strength: 0.875,
                        timestep_start: None,
                        timestep_end: None,
                    },
                    conditions: vec![(
                        "c_crossattn".into(),
                        Condition::tensor(
                            CKind::CrossAttention,
                            Tensor::full([1, 3, 2], d["value"].as_f64().unwrap(), (dtype, dev)),
                        ),
                    )],
                    hooks: d["hook"].as_u64().map(|n| hooks[n as usize].clone()),
                    control: d["control"].as_u64().map(|n| controls[n as usize].clone()),
                    gligen: d["gligen"].as_str().map(|kind| {
                        Rc::new(Gligen {
                            kind: kind.into(),
                            log: log.clone(),
                        }) as Rc<dyn b::Gligen>
                    }),
                    uuid: V::Text(d["uuid"].as_str().unwrap().into()),
                    default: d.as_object().unwrap().contains_key("default"),
                });
            }
            groups.push(Some(rows));
        }
        let refs = groups
            .iter()
            .map(|rows| rows.as_deref())
            .collect::<Vec<_>>();
        let input = data(&[1, 2, 12, 16], dev, dtype, 3);
        let time = Tensor::from_slice(&[0.75f32]).to_device(dev);
        let options = Options::from_value(V::Dict(vec![(
            "transformer_options".into(),
            V::Dict(vec![
                (
                    "nested".into(),
                    V::Dict(vec![("a".into(), V::List(vec![V::Int(3)]))]),
                ),
                (
                    "patches".into(),
                    V::Dict(vec![(
                        "shared_patch".into(),
                        V::List(vec![V::Text("own-shared".into())]),
                    )]),
                ),
            ]),
        )]));
        let mut runtime = Runtime {
            log: log.clone(),
            hooks,
        };
        let out = b::calculate(&mut runtime, &refs, &input, &time, &options)
            .unwrap_or_else(|e| panic!("{e}: {c}"));
        for (index, tensor) in out.into_iter().enumerate() {
            check(
                &V::tensor(tensor),
                &c["output"][index],
                dev,
                &format!("Conditioning batch {} {}", c["mode"], c["free"]),
            );
        }
        assert_eq!(log.index.get(), c["events"].as_array().unwrap().len());
    }
}
