use super::{
    binding_tests::{check, reference_error, restore},
    device, kind, kitchen,
};
use imagine_inference::{
    context_windows as c,
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
};
use serde_json::{Value, json};
use tch::{Device, Kind};
fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn windows(v: Vec<Vec<i64>>) -> V {
    V::List(
        v.into_iter()
            .map(|v| V::List(v.into_iter().map(V::Int).collect()))
            .collect(),
    )
}
struct DefaultKind(imagine_inference::native::Kitchen, Kind);
impl Drop for DefaultKind {
    fn drop(&mut self) {
        self.0.set_default_dtype(self.1).unwrap();
    }
}
#[test]
fn original_context_schedules_fusion_weights_dimension_shapes_and_private_freenoise_rng_match() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let _kind = DefaultKind(rt.native.clone(), rt.native.default_dtype().unwrap());
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-context-windows.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5637);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = p["device"].as_str().map(device).unwrap_or(Device::Cpu);
        let default = p["default"].as_str().map(kind).unwrap_or(Kind::Float);
        rt.native.set_default_dtype(default).unwrap();
        let mut warning = vec![];
        let mut infos = vec![];
        let mut post = None;
        let name = case["name"].as_str().unwrap();
        let outcome: Result<V, Error> = match name {
            "schedule" => c::Schedule::from_name(p["schedule"].as_str().unwrap())
                .and_then(|s| {
                    s.windows(
                        p["frames"].as_i64().unwrap(),
                        c::ScheduleSettings {
                            length: p["length"].as_i64().unwrap(),
                            overlap: p["overlap"].as_i64().unwrap(),
                            stride: p["stride"].as_i64().unwrap(),
                            closed_loop: p["closed"].as_bool().unwrap(),
                            step: p["step"].as_u64().unwrap(),
                        },
                        &mut |message| {
                            warning.push(json!({"category":"RuntimeWarning","message":message}));
                            Ok(())
                        },
                    )
                })
                .map(windows),
            "schedule-name" => c::Schedule::from_name(p["value"].as_str().unwrap()).map(|s| {
                V::Tuple(vec![
                    V::Text(s.name().into()),
                    V::Text(format!(
                        "create_windows_{}",
                        match s {
                            c::Schedule::UniformLooped => "uniform_looped",
                            c::Schedule::UniformStandard => "uniform_standard",
                            c::Schedule::StaticStandard => "static_standard",
                            c::Schedule::Batched => "batched",
                        }
                    )),
                ])
            }),
            "fuse-name" => c::Fuse::from_name(p["value"].as_str().unwrap()).map(|f| {
                V::Tuple(vec![
                    V::Text(f.name().into()),
                    V::Text(format!(
                        "create_weights_{}",
                        match f {
                            c::Fuse::Flat => "flat",
                            c::Fuse::Pyramid | c::Fuse::Relative => "pyramid",
                            c::Fuse::OverlapLinear => "overlap_linear",
                        }
                    )),
                ])
            }),
            "halving" => Ok(V::Float(c::ordered_halving(p["value"].as_u64().unwrap()))),
            "default" => c::default_windows(p["frames"].as_i64().unwrap()).map(windows),
            "roll" => c::does_roll_over(&ints(&p["window"]), p["frames"].as_i64().unwrap())
                .map(|(a, b)| V::Tuple(vec![V::Bool(a), V::Int(b)])),
            "shift-start" | "shift-end" => {
                let mut v = ints(&p["window"]);
                let r = if name == "shift-start" {
                    c::shift_to_start(&mut v, p["frames"].as_i64().unwrap())
                } else {
                    c::shift_to_end(&mut v, p["frames"].as_i64().unwrap())
                };
                post = Some(V::List(v.into_iter().map(V::Int).collect()));
                r.map(|_| V::None)
            }
            "missing" => c::missing_indexes(
                &p["windows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(ints)
                    .collect::<Vec<_>>(),
                p["frames"].as_i64().unwrap(),
            )
            .map(|v| V::List(v.into_iter().map(V::Int).collect())),
            "weights" => c::Fuse::from_name(p["method"].as_str().unwrap()).and_then(|f| {
                f.weights(
                    &rt.native,
                    p["length"].as_i64().unwrap(),
                    p["frames"].as_i64().unwrap_or(0),
                    &p["indices"]
                        .as_array()
                        .map(|_| ints(&p["indices"]))
                        .unwrap_or_default(),
                    p["overlap"].as_i64().unwrap_or(0),
                )
            }),
            "shape" => {
                let x = restore(&case["initial"], dev).as_tensor().unwrap();
                c::shape_for_dim(&x, p["dim"].as_i64().unwrap())
                    .map(|v| V::List(v.into_iter().map(V::Int).collect()))
            }
            "match-weights" => {
                let x = restore(&case["initial"]["input"], dev).as_tensor().unwrap();
                let weights = restore(&case["initial"]["weights"], dev);
                c::match_weights_to_dim(
                    &rt,
                    &weights,
                    &x,
                    p["dim"].as_i64().unwrap(),
                    p["out_device"].as_str().map(device),
                )
                .map(V::tensor)
            }
            "freenoise" => {
                let noise = restore(&case["initial"], dev);
                tch::manual_seed(17);
                let cpu = native::global_rng_state(Device::Cpu).unwrap();
                let gpu = native::global_rng_state(Device::Cuda(0)).unwrap();
                let result = c::apply_freenoise(
                    &rt.native,
                    &noise,
                    p["dim"].as_i64().unwrap(),
                    p["length"].as_i64().unwrap(),
                    p["overlap"].as_i64().unwrap(),
                    p["seed"].as_u64().unwrap(),
                    &mut |message| {
                        infos.push(message.to_owned());
                        Ok(())
                    },
                );
                assert!(cpu.equal(&native::global_rng_state(Device::Cpu).unwrap()));
                assert!(gpu.equal(&native::global_rng_state(Device::Cuda(0)).unwrap()));
                if let Ok(v) = &result {
                    assert!(v.same_object(&noise));
                }
                post = Some(noise);
                result
            }
            _ => panic!("own case {name}"),
        };
        let note = format!("context {index} {name} {p}");
        match outcome {
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: {e}");
                reference_error(e, case);
            }
            Ok(v) => {
                assert!(
                    !case["error"].is_string(),
                    "{note}: expected {} {}",
                    case["error"],
                    case["message"]
                );
                check(&v, &case["output"], dev, &note);
            }
        }
        if let Some(post) = post {
            check(&post, &case["post"], dev, &note);
        }
        assert_eq!(json!(warning), case["warnings"], "warnings {note}");
        assert_eq!(json!(infos), case["infos"], "infos {note}");
    }
}
