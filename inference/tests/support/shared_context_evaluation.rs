use super::{
    binding_tests::{check, reference_error, restore},
    device, kind, kitchen,
    shared_context_state_tests::{data, l, snapshot},
};
use imagine_inference::{
    conditioning::{Condition, Kind as CK},
    context_windows::{
        self as c, BuiltinConditionAttributes, ConditionAttributes, ContextBatch, ContextBindings,
        ContextCall, ContextCallbackFrame, ContextModel, EvaluationLocals, EvaluationRequest,
        Handler, IndexWindow, LatentShape, LatentShapes, ModalityIndices, ResizeModel,
        ResizeModelSlot, ResizingBindings, WindowHandle, WindowList, WindowStateModel,
    },
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options as O,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use tch::{Device, Kind, Tensor};
fn none() -> O {
    O::Atom(V::None)
}
fn list(v: Vec<O>) -> O {
    O::List(l(v))
}
fn dict(v: Vec<(&str, O)>) -> O {
    O::dictionary(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn constant(v: O) -> O {
    O::Object(Rc::new(Condition::from_options(CK::Constant, v)))
}
fn plain(v: &V) -> Value {
    match v {
        V::Int(n) => json!(n),
        V::Float(n) => json!(n),
        V::Bool(v) => json!(v),
        V::None => Value::Null,
        V::List(v) | V::Tuple(v) => json!(v.iter().map(plain).collect::<Vec<_>>()),
        V::Dict(v) => Value::Object(v.iter().map(|(k, v)| (k.clone(), plain(v))).collect()),
        _ => panic!("unexpected plain window metadata"),
    }
}
fn window_json(window: &WindowHandle) -> Value {
    plain(&snapshot(window))
}
fn check_value(v: &V, e: &Value, note: &str) {
    match v {
        V::NamedTuple(v) => {
            assert_eq!(v.class.name, "Size", "{note}");
            assert_eq!(e["type"], "Size", "{note}");
            assert_eq!(
                v.fields.len(),
                e["items"].as_array().unwrap().len(),
                "{note}"
            );
            for ((_, v), e) in v.fields.iter().zip(e["items"].as_array().unwrap()) {
                check_value(v, e, note);
            }
        }
        V::List(items) | V::Tuple(items) => {
            assert_eq!(
                e["type"],
                if matches!(v, V::List(_)) {
                    "list"
                } else {
                    "tuple"
                },
                "{note}"
            );
            let expected = e["items"].as_array().unwrap();
            assert_eq!(items.len(), expected.len(), "{note}");
            for (v, e) in items.iter().zip(expected) {
                check_value(v, e, note);
            }
        }
        V::Dict(items) => {
            assert_eq!(e["type"], "dict", "{note}");
            let expected = e["items"].as_array().unwrap();
            assert_eq!(items.len(), expected.len(), "{note}");
            for ((k, v), e) in items.iter().zip(expected) {
                assert_eq!(k, e[0].as_str().unwrap(), "{note}");
                check_value(v, &e[1], note);
            }
        }
        _ => check(v, e, Device::Cpu, note),
    }
}
pub(super) fn check_options(v: &O, e: &Value, note: &str) {
    match v {
        O::Object(v) if v.is::<Condition>() => {
            let c = v.downcast_ref::<Condition>().unwrap();
            assert_eq!(e["type"], "condition", "{note}");
            let name = match c.kind {
                CK::Regular => "CONDRegular",
                CK::NoiseShape => "CONDNoiseShape",
                CK::CrossAttention => "CONDCrossAttn",
                CK::Constant => "CONDConstant",
                CK::List => "CONDList",
            };
            assert_eq!(e["name"], name, "{note}");
            check_options(&c.source_value(), &e["value"], note);
        }
        O::Dict(v) => {
            assert_eq!(e["type"], "dict", "{note}");
            let v = v.borrow();
            let expected = e["items"].as_array().unwrap();
            assert_eq!(v.len(), expected.len(), "{note}");
            for ((k, v), e) in v.iter().zip(expected) {
                assert_eq!(k, e[0].as_str().unwrap(), "{note}");
                check_options(v, &e[1], note);
            }
        }
        O::List(v) => {
            assert_eq!(e["type"], "list", "{note}");
            let v = v.borrow();
            let expected = e["items"].as_array().unwrap();
            assert_eq!(v.len(), expected.len(), "{note}");
            for (v, e) in v.iter().zip(expected) {
                check_options(v, e, note);
            }
        }
        O::Tuple(v) => {
            assert_eq!(e["type"], "tuple", "{note}");
            let expected = e["items"].as_array().unwrap();
            assert_eq!(v.len(), expected.len(), "{note}");
            for (v, e) in v.iter().zip(expected) {
                check_options(v, e, note);
            }
        }
        _ => check_value(&v.snapshot().unwrap(), e, note),
    }
}
struct Shared {
    case: Value,
    position: Cell<usize>,
    interrupts: Cell<usize>,
    infos: RefCell<Vec<String>>,
    seen: WindowList<WindowHandle>,
    h: Weak<Handler>,
    model: RefCell<Option<Weak<dyn ContextModel>>>,
    active_window: RefCell<Option<Weak<RefCell<IndexWindow>>>>,
    input: V,
    time: V,
    conditions: O,
    options: O,
    windows: WindowList<(i64, WindowHandle)>,
    shapes: O,
    note: String,
    dev: Device,
}
impl Shared {
    fn mode(&self) -> &str {
        self.case["params"]["mode"].as_str().unwrap()
    }
    fn next(&self, tag: &str) -> Value {
        let i = self.position.get();
        let expected = self.case["events"]
            .as_array()
            .unwrap()
            .get(i)
            .unwrap_or_else(|| panic!("{} unexpected {tag} after event {i}", self.note))
            .clone();
        assert_eq!(expected["tag"], tag, "{} event {i}", self.note);
        self.position.set(i + 1);
        expected
    }
}
struct Model(Rc<Shared>);
impl WindowStateModel for Model {
    fn temporal_downscale_ratio(&self) -> Result<i64, Error> {
        Ok(2)
    }
}
impl ResizeModel for Model {
    fn resize_cond_for_context_window(
        &self,
        key: &str,
        _: &O,
        _: &WindowHandle,
        _: &V,
        _: Option<Device>,
        retain: WindowList<i64>,
    ) -> Result<O, Error> {
        let s = &self.0;
        let e = s.next("resize");
        assert_eq!(e["key"], key, "{}", s.note);
        assert_eq!(
            e["same_retain"],
            Rc::ptr_eq(
                &retain,
                &s.h.upgrade().unwrap().settings.borrow().condition_retain
            ),
            "{}",
            s.note
        );
        Ok(none())
    }
}
impl ContextModel for Model {
    fn class_name(&self) -> &str {
        "Model"
    }
    fn map_context_window_to_modalities(
        &self,
        indices: WindowList<i64>,
        shapes: LatentShapes,
        dim: i64,
    ) -> Result<ModalityIndices, Error> {
        let s = &self.0;
        let e = s.next("map");
        let original = LatentShapes::from_options(&s.shapes)?;
        let entries = shapes.entries()?;
        assert_eq!(json!(*indices.borrow()), e["indices"], "{}", s.note);
        assert_eq!(
            json!(
                entries
                    .iter()
                    .map(LatentShape::values)
                    .collect::<Result<Vec<_>, _>>()?
            ),
            e["shapes"],
            "{}",
            s.note
        );
        assert_eq!(
            json!(
                entries
                    .iter()
                    .map(LatentShape::class_name)
                    .collect::<Vec<_>>()
            ),
            e["types"],
            "{}",
            s.note
        );
        assert_eq!(
            shapes.same_object(&original),
            e["same_shapes"],
            "{}",
            s.note
        );
        assert_eq!(
            json!(
                entries
                    .iter()
                    .zip(original.entries()?.iter())
                    .map(|(a, b)| a.same_object(b))
                    .collect::<Vec<_>>()
            ),
            e["same_shape_objects"],
            "{}",
            s.note
        );
        if s.mode() == "map-attribute" {
            return Err(Error::ReferenceException {
                class: "AttributeError",
                message: "own evaluation mapping attribute failure".into(),
            });
        }
        if s.mode() == "map-error" {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own evaluation mapping failure".into(),
            });
        }
        let primary = shapes.get(0)?.dimension(dim)?;
        let secondary = shapes.get(1)?.dimension(dim)?;
        let child = indices
            .borrow()
            .iter()
            .map(|i| {
                ((i128::from(*i) * i128::from(secondary)) / i128::from(primary))
                    .min(i128::from(secondary - 1)) as i64
            })
            .collect();
        Ok(l(vec![indices, l(child)]))
    }
}
struct Bindings(Rc<Shared>);
impl ConditionAttributes for Bindings {
    fn cond(&self, v: &O) -> Result<O, Error> {
        BuiltinConditionAttributes.cond(v)
    }
}
impl ResizingBindings for Bindings {
    fn prepare_control_objects(&self, _: &O, _: Option<Device>) -> Result<O, Error> {
        panic!("unexpected control binding in bounded evaluation fixture")
    }
    fn info(&self, message: &str) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next("info-message");
        assert_eq!(e["message"], message, "{}", s.note);
        s.infos.borrow_mut().push(message.into());
        if s.mode() == "info-error" {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own evaluation info failure".into(),
            });
        }
        Ok(())
    }
}
impl ContextBindings for Bindings {
    fn evaluation_locals(&self, locals: &EvaluationLocals, tag: &str) -> Result<(), Error> {
        if tag == "prepare_window" {
            return Ok(());
        }
        let s = &self.0;
        if tag == "model" {
            *s.active_window.borrow_mut() = locals.window.as_ref().map(Rc::downgrade);
        }
        let e = s.next(tag);
        let mut pairs = vec![("x".into(), locals.primary.clone())];
        if let Some(v) = &locals.sliced {
            for (i, v) in v.iter().enumerate() {
                pairs.push((format!("sliced.{i}"), v.clone()));
            }
        }
        if let Some(v) = &locals.sub_input {
            pairs.push(("sub_x".into(), v.clone()));
        }
        if let Some(v) = &locals.sub_timestep {
            pairs.push(("sub_timestep".into(), v.clone()));
        }
        if let Some(v) = &locals.model_outputs {
            for (i, v) in v.borrow().iter().enumerate() {
                pairs.push((format!("sub_conds_out.{i}"), v.clone()));
            }
        }
        if let Some(v) = &locals.outputs {
            for (i, row) in v.borrow().iter().enumerate() {
                for (j, v) in row.borrow().iter().enumerate() {
                    pairs.push((format!("out_per_modality.{i}.{j}"), v.clone()));
                }
            }
        }
        if let Some(v) = &locals.last_primary_output {
            pairs.push(("t".into(), v.clone()));
        }
        assert_eq!(
            json!(pairs.iter().map(|(k, _)| k).collect::<Vec<_>>()),
            e["names"],
            "{} {tag}",
            s.note
        );
        for (k, v) in &pairs {
            check(v, &e["tensors"][k], s.dev, &format!("{} {tag} {k}", s.note));
        }
        assert_eq!(
            json!(
                pairs
                    .iter()
                    .map(|(_, a)| pairs
                        .iter()
                        .map(|(_, b)| a.same_object(b))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            e["same_objects"],
            "{} {tag}",
            s.note
        );
        assert_eq!(
            json!(
                pairs
                    .iter()
                    .map(|(_, a)| pairs
                        .iter()
                        .map(
                            |(_, b)| native::storage_info(&a.as_tensor().unwrap()).unwrap().0
                                == native::storage_info(&b.as_tensor().unwrap()).unwrap().0
                        )
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            ),
            e["same_storages"],
            "{} {tag}",
            s.note
        );
        Ok(())
    }
    fn throw_if_interrupted(&self) -> Result<(), Error> {
        let s = &self.0;
        let n = s.interrupts.get() + 1;
        s.interrupts.set(n);
        if s.mode() == "interrupt-first" && n == 1 || s.mode() == "interrupt-second" && n == 2 {
            return Err(Error::ReferenceException {
                class: "InterruptProcessingException",
                message: String::new(),
            });
        }
        Ok(())
    }
    fn callback(&self, _: &O, frame: ContextCallbackFrame<'_>) -> Result<(), Error> {
        let ContextCallbackFrame::Evaluate {
            call,
            window_index,
            window,
            device,
            first_device,
        } = frame
        else {
            panic!("unexpected callback tag")
        };
        let s = &self.0;
        let e = s.next("callback-args");
        s.seen.borrow_mut().push(window.clone());
        assert_eq!(
            Rc::ptr_eq(call.handler, &s.h.upgrade().unwrap()),
            e["same_handler"],
            "{}",
            s.note
        );
        assert_eq!(
            Rc::ptr_eq(
                call.model,
                &s.model.borrow().as_ref().unwrap().upgrade().unwrap()
            ),
            e["same_model"],
            "{}",
            s.note
        );
        assert_eq!(
            call.input.same_object(&s.input),
            e["same_input"],
            "{}",
            s.note
        );
        assert_eq!(
            call.conditions.same_object(&s.conditions),
            e["same_conditions"],
            "{}",
            s.note
        );
        assert_eq!(
            call.timestep.same_object(&s.time),
            e["same_time"],
            "{}",
            s.note
        );
        assert_eq!(
            call.options.same_object(&s.options),
            e["same_options"],
            "{}",
            s.note
        );
        assert_eq!(
            call.options.same_object(&s.options),
            e["same_options_again"],
            "{}",
            s.note
        );
        assert_eq!(json!(window_index), e["index"], "{}", s.note);
        assert_eq!(window_json(window), e["window"], "{}", s.note);
        assert_eq!(
            json!(device.map(|v| match v {
                Device::Cpu => "cpu".to_string(),
                Device::Cuda(i) => format!("cuda:{i}"),
                _ => panic!("unexpected device"),
            })),
            e["device"],
            "{}",
            s.note
        );
        assert_eq!(
            json!(first_device.map(|v| match v {
                Device::Cpu => "cpu".to_string(),
                _ => panic!("unexpected first device"),
            })),
            e["first_device"],
            "{}",
            s.note
        );
        if s.mode() == "callback-error" {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own evaluation callback failure".into(),
            });
        }
        let h = s.h.upgrade().unwrap();
        match s.mode() {
            "callback-mutate" => {
                h.settings.borrow_mut().length += 1;
                s.options.set("own_callback", O::Atom(V::Int(11)))?;
            }
            "callback-clear-windows" => s.windows.borrow_mut().clear(),
            "callback-clear-model" => *h.model.borrow_mut() = ResizeModelSlot::None,
            "callback-remove-options" => {
                s.options.pop("transformer_options")?;
            }
            "callback-bad-dim" => h.settings.borrow_mut().dim = 7,
            _ => {}
        }
        Ok(())
    }
    fn calc_cond_batch(&self, call: &ContextBatch<'_>) -> Result<WindowList<V>, Error> {
        let s = &self.0;
        let e = s.next("batch-args");
        assert_eq!(
            Rc::ptr_eq(
                call.model,
                &s.model.borrow().as_ref().unwrap().upgrade().unwrap()
            ),
            e["same_model"],
            "{}",
            s.note
        );
        assert_eq!(
            call.options.same_object(&s.options),
            e["same_options"],
            "{}",
            s.note
        );
        let active = s
            .active_window
            .borrow()
            .as_ref()
            .unwrap()
            .upgrade()
            .unwrap();
        assert_eq!(
            call.options
                .get("transformer_options")?
                .unwrap()
                .get("context_window")?
                .unwrap()
                .same_object(&O::Object(active)),
            e["same_window"],
            "{}",
            s.note
        );
        check(call.input, &e["input"], s.dev, &s.note);
        check(call.timestep, &e["time"], s.dev, &s.note);
        check_options(call.conditions, &e["conditions"], &s.note);
        if s.mode() == "model-error" {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own evaluation model failure".into(),
            });
        }
        if s.mode() == "model-mutate" {
            let _ = call.input.as_tensor()?.f_add_scalar_(0.25)?;
            call.options.set("own_model", O::Atom(V::Int(13)))?;
        }
        let O::List(conditions) = call.conditions else {
            panic!("own condition rows")
        };
        let mut outputs = vec![];
        for (ci, condition) in conditions.borrow().iter().enumerate() {
            let input = call.input.as_tensor()?;
            let output = if matches!(condition, O::Atom(V::None)) {
                input.f_zeros_like()?
            } else {
                let O::List(entries) = condition else {
                    panic!("own condition list")
                };
                let marker = if let Some(entry) = entries.borrow().first() {
                    match entry.get("marker")?.unwrap() {
                        O::Atom(V::Int(n)) => n,
                        _ => panic!("own marker"),
                    }
                } else {
                    0
                };
                let input = if s.mode() == "model-promote" {
                    input.f_to_kind(Kind::Double)?
                } else {
                    input
                };
                input
                    .f_mul_scalar(0.125 + 0.0625 * marker as f64)?
                    .f_add_scalar(0.03125 * ci as f64)?
            };
            outputs.push(V::tensor(output));
        }
        Ok(l(outputs))
    }
}
#[test]
fn whole_context_evaluation_preserves_modalities_guides_anchor_order_conditions_callbacks_and_live_tensor_locals()
 {
    let _rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-evaluation.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 640);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = device(p["device"].as_str().unwrap());
        let dtype = kind(p["dtype"].as_str().unwrap());
        let multi = p["multi"] == true;
        let guides = p["guides"] == true;
        let frames = if guides { 9 } else { 12 };
        let x = restore(&case["input"], dev);
        let time = restore(&case["time"], dev);
        let mut sizes = vec![vec![1, 1, 12, 2, 2]];
        if multi {
            sizes.push(vec![1, 1, 6, 1, 1]);
        }
        let shapes = list(
            sizes
                .into_iter()
                .map(|v| LatentShape::Size(Rc::new(v)).as_options())
                .collect(),
        );
        let entries = list(vec![
            dict(vec![
                (
                    "latent_shape",
                    O::Tuple(Rc::new(
                        vec![2, 1, 1]
                            .into_iter()
                            .map(|n| O::Atom(V::Int(n)))
                            .collect(),
                    )),
                ),
                ("pre_filter_count", O::Atom(V::Int(1))),
            ]),
            dict(vec![
                (
                    "latent_shape",
                    O::Tuple(Rc::new(
                        vec![1, 2, 2]
                            .into_iter()
                            .map(|n| O::Atom(V::Int(n)))
                            .collect(),
                    )),
                ),
                ("pre_filter_count", O::Atom(V::Int(1))),
            ]),
        ]);
        let kf = Tensor::zeros([1, 3, 2, 2], (Kind::Int64, dev));
        let times = Tensor::from_slice(&[2i64, 12]).to_device(dev);
        kf.get(0).get(0).select(1, 0).copy_(&times);
        let kf = V::tensor(kf);
        let entry = |marker: i64| {
            let inner = dict(vec![
                (
                    "regular",
                    O::Object(Rc::new(Condition::tensor(
                        CK::Regular,
                        data(&[1, 1, frames, 2, 2], dev, dtype, marker + 3)
                            .as_tensor()
                            .unwrap(),
                    ))),
                ),
                ("num_video_frames", constant(O::Atom(V::Int(12)))),
            ]);
            if multi {
                inner
                    .set("latent_shapes", constant(shapes.clone()))
                    .unwrap();
            }
            if guides {
                inner
                    .set("guide_attention_entries", constant(entries.clone()))
                    .unwrap();
                inner
                    .set("keyframe_idxs", constant(O::Atom(kf.clone())))
                    .unwrap();
            }
            dict(vec![
                (
                    "top",
                    O::Atom(data(&[1, 1, frames, 2, 2], dev, dtype, marker + 1)),
                ),
                ("model_conds", inner),
                ("marker", O::Atom(V::Int(marker))),
            ])
        };
        let conditions = if p["batch"] == "empty" {
            list(vec![])
        } else {
            list(vec![
                if p["batch"] == "none" {
                    none()
                } else {
                    list(vec![entry(1), entry(2)])
                },
                list(vec![entry(5)]),
            ])
        };
        let options = dict(vec![(
            "transformer_options",
            dict(vec![("own", O::Atom(V::Int(7)))]),
        )]);
        let h = Rc::new(Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat));
        {
            let mut settings = h.settings.borrow_mut();
            settings.dim = 2;
            settings.length = 3;
            settings.overlap = 1;
            settings.causal_window_fix = p["causal"] == true;
            settings.split_conditions = p["split"] == true;
            settings.condition_retain = l(if p["retains"] == true {
                vec![0]
            } else {
                vec![]
            });
            settings.latent_retain = l(if p["retains"] == true {
                vec![0]
            } else {
                vec![]
            });
        }
        let windows = l((0..3)
            .map(|i| {
                (
                    i,
                    IndexWindow::new(l((i * 3..i * 3 + 3).collect()), 2, frames, None, 1).unwrap(),
                )
            })
            .collect());
        let shared = Rc::new(Shared {
            case: case.clone(),
            position: Cell::new(0),
            interrupts: Cell::new(0),
            infos: RefCell::new(vec![]),
            seen: l(vec![]),
            h: Rc::downgrade(&h),
            model: RefCell::new(None),
            active_window: RefCell::new(None),
            input: x.clone(),
            time: time.clone(),
            conditions: conditions.clone(),
            options: options.clone(),
            windows: windows.clone(),
            shapes,
            note: format!("context evaluation {index} {p}"),
            dev,
        });
        let model: Rc<dyn ContextModel> = Rc::new(Model(shared.clone()));
        *shared.model.borrow_mut() = Some(Rc::downgrade(&model));
        *h.model.borrow_mut() = ResizeModelSlot::Bound(model.clone());
        if shared.mode().starts_with("callback-") {
            h.callbacks.borrow_mut().add(
                "evaluate_context_windows",
                Some("own"),
                Rc::new(O::Object(Rc::new("eval-callback"))),
            );
        }
        let bindings = Bindings(shared.clone());
        let state = h
            .build_window_state(&x, &conditions, &bindings, model.as_ref())
            .unwrap();
        let out = Handler::evaluate_context_windows(
            &bindings,
            EvaluationRequest {
                call: ContextCall {
                    handler: &h,
                    model: &model,
                    input: &x,
                    conditions: &conditions,
                    timestep: &time,
                    options: &options,
                },
                windows: &windows,
                state: &state,
                total_windows: p["total"].as_i64(),
                device: p["target"].as_str().map(device),
                first_device: Some(Device::Cpu),
            },
        );
        let rows = match out {
            Ok(v) => {
                assert_eq!(case["ok"], true, "{}", shared.note);
                v
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{}: {e}", shared.note);
                reference_error(e, case);
                l(vec![])
            }
        };
        assert_eq!(
            shared.position.get(),
            case["events"].as_array().unwrap().len(),
            "{}",
            shared.note
        );
        assert_eq!(
            json!(*shared.infos.borrow()),
            case["infos"],
            "{}",
            shared.note
        );
        let expected = case["results"].as_array().unwrap();
        assert_eq!(rows.borrow().len(), expected.len(), "{}", shared.note);
        let option_window = options
            .get("transformer_options")
            .unwrap()
            .and_then(|v| v.get("context_window").unwrap());
        for (row, e) in rows.borrow().iter().zip(expected) {
            assert_eq!(json!(row.window_index), e["index"], "{}", shared.note);
            check(
                &V::List(
                    row.outputs
                        .borrow()
                        .iter()
                        .map(|row| V::List(row.borrow().clone()))
                        .collect(),
                ),
                &e["outputs"],
                dev,
                &shared.note,
            );
            check_options(&row.conditions, &e["conditions"], &shared.note);
            assert_eq!(window_json(&row.window), e["window"], "{}", shared.note);
            assert_eq!(
                option_window
                    .as_ref()
                    .is_some_and(|v| v.same_object(&O::Object(row.window.clone()))),
                e["same_option_window"],
                "{}",
                shared.note
            );
        }
        let option_json = option_window
            .map(|v| {
                let O::Object(v) = v else {
                    panic!("own context window")
                };
                let v = v.downcast::<RefCell<c::IndexWindow>>().unwrap();
                window_json(&v)
            })
            .unwrap_or(Value::Null);
        assert_eq!(
            json!({"length":h.settings.borrow().length,"dim":h.settings.borrow().dim,"windows":windows.borrow().iter().map(|(i,w)|json!([i,window_json(w)])).collect::<Vec<_>>(),"seen":shared.seen.borrow().iter().map(window_json).collect::<Vec<_>>(),"option_window":option_json,"own_callback":options.get("own_callback").unwrap().map(|v|plain(&v.snapshot().unwrap())),"own_model":options.get("own_model").unwrap().map(|v|plain(&v.snapshot().unwrap()))}),
            case["post"],
            "{}",
            shared.note
        );
    }
}
