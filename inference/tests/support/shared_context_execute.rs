use super::{
    binding_tests::{check, reference_error, restore},
    device, kind, kitchen,
    shared_context_evaluation_tests::check_options,
    shared_context_state_tests::{data, l},
};
use imagine_inference::{
    conditioning::{Condition, Kind as CK},
    context_windows::{
        self as c, BuiltinConditionAttributes, ConditionAttributes, ContextBatch, ContextBindings,
        ContextCall, ContextCallbackFrame, ContextModel, ExecutionBindings, ExecutionLocals,
        Handler, LatentShape, LatentShapes, ModalityIndices, ResizeModel, ResizeModelSlot,
        ResizingBindings, WindowHandle, WindowList, WindowStateModel,
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
fn list(v: Vec<O>) -> O {
    O::List(l(v))
}
fn dict(v: Vec<(&str, O)>) -> O {
    O::dictionary(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn constant(v: O) -> O {
    O::Object(Rc::new(Condition::from_options(CK::Constant, v)))
}
fn id<T>(v: &Rc<T>) -> usize {
    Rc::as_ptr(v) as usize
}
fn tensor_id(v: &V) -> usize {
    let V::Tensor(v) = v else {
        panic!("own tensor")
    };
    std::sync::Arc::as_ptr(v) as usize
}
fn option_list_id(v: &O) -> usize {
    let O::List(v) = v else { panic!("own list") };
    id(v)
}
#[derive(Default)]
struct ObservedCombine {
    input: usize,
    outputs: usize,
    conditions: usize,
    window: usize,
    accum: usize,
    counts: usize,
    biases: usize,
    index: i64,
    total: usize,
    modality: usize,
}
type WeakAccum = Weak<RefCell<Vec<WindowList<V>>>>;
struct Shared {
    case: Value,
    position: Cell<usize>,
    interrupts: Cell<usize>,
    cleanup: Cell<usize>,
    infos: RefCell<Vec<String>>,
    warnings: RefCell<Vec<Value>>,
    h: Weak<Handler>,
    model: RefCell<Option<Weak<dyn ContextModel>>>,
    input: V,
    time: V,
    conditions: O,
    options: O,
    dev: Device,
    note: String,
    observed: RefCell<ObservedCombine>,
    accum: RefCell<Option<WeakAccum>>,
}
impl Shared {
    fn mode(&self) -> &str {
        self.case["params"]["mode"].as_str().unwrap()
    }
    fn next(&self, tag: &str) -> Value {
        let i = self.position.get();
        let e = self.case["events"]
            .as_array()
            .unwrap()
            .get(i)
            .unwrap_or_else(|| panic!("{} unexpected {tag} at {i}", self.note))
            .clone();
        assert_eq!(e["tag"], tag, "{} event {i}", self.note);
        self.position.set(i + 1);
        e
    }
    fn fail(&self, message: &str) -> Error {
        Error::ReferenceException {
            class: "ValueError",
            message: message.into(),
        }
    }
    fn same_call(&self, call: &ContextCall<'_>) -> Vec<bool> {
        vec![
            Rc::ptr_eq(call.handler, &self.h.upgrade().unwrap()),
            Rc::ptr_eq(
                call.model,
                &self.model.borrow().as_ref().unwrap().upgrade().unwrap(),
            ),
            call.input.same_object(&self.input),
            call.conditions.same_object(&self.conditions),
            call.timestep.same_object(&self.time),
            call.options.same_object(&self.options),
        ]
    }
    fn check_callback(&self, expected: &Value, phase: &str, same: Vec<bool>) {
        assert_eq!(expected["phase"], phase, "{}", self.note);
        assert_eq!(expected["same"], json!(same), "{}", self.note);
    }
}
struct Model;
impl WindowStateModel for Model {
    fn temporal_downscale_ratio(&self) -> Result<i64, Error> {
        Ok(2)
    }
}
impl ResizeModel for Model {
    fn resize_cond_for_context_window(
        &self,
        _: &str,
        _: &O,
        _: &WindowHandle,
        _: &V,
        _: Option<Device>,
        _: WindowList<i64>,
    ) -> Result<O, Error> {
        Ok(O::Atom(V::None))
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
        panic!("unexpected control in bounded execute fixture")
    }
    fn info(&self, message: &str) -> Result<(), Error> {
        self.0.infos.borrow_mut().push(message.into());
        Ok(())
    }
}
impl ContextBindings for Bindings {
    fn throw_if_interrupted(&self) -> Result<(), Error> {
        let s = &self.0;
        let n = s.interrupts.get() + 1;
        s.interrupts.set(n);
        if s.mode() == "interrupt-second" && n == 2 {
            return Err(Error::ReferenceException {
                class: "InterruptProcessingException",
                message: String::new(),
            });
        }
        Ok(())
    }
    fn callback(&self, _: &O, frame: ContextCallbackFrame<'_>) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next("callback-args");
        let h = s.h.upgrade().unwrap();
        match frame {
            ContextCallbackFrame::ExecuteStart { call } => {
                let same = s.same_call(call);
                s.check_callback(&e, "start-callback", same);
                match s.mode() {
                    "start-error" => return Err(s.fail("own execute start failure")),
                    "start-clear-conditions" => {
                        let O::List(v) = &s.conditions else {
                            unreachable!()
                        };
                        v.borrow_mut().clear();
                    }
                    "start-change-fuse" => {
                        let fuse = h.settings.borrow().fuse;
                        h.settings.borrow_mut().fuse = if matches!(fuse, c::Fuse::Relative) {
                            c::Fuse::Flat
                        } else {
                            c::Fuse::Relative
                        };
                    }
                    "start-remove-options" => {
                        s.options.pop("transformer_options")?;
                    }
                    _ => {}
                }
            }
            ContextCallbackFrame::Evaluate {
                call,
                device,
                first_device,
                ..
            } => {
                let mut same = s.same_call(call);
                same.extend([
                    call.options.same_object(&s.options),
                    device.is_none(),
                    first_device.is_none(),
                ]);
                s.check_callback(&e, "evaluate-callback", same);
                if s.mode() == "evaluate-error" {
                    return Err(s.fail("own execute evaluation failure"));
                }
                if s.mode() == "evaluate-clear-model" {
                    *h.model.borrow_mut() = ResizeModelSlot::None;
                }
            }
            ContextCallbackFrame::Combine {
                call,
                frame,
                conditions,
                window_index,
                total_windows,
            } => {
                let o = s.observed.borrow();
                let same = vec![
                    Rc::ptr_eq(call.handler, &h),
                    tensor_id(frame.input) == o.input,
                    id(frame.outputs) == o.outputs,
                    option_list_id(conditions) == o.conditions,
                    id(frame.window) == o.window,
                    window_index == o.index,
                    total_windows == o.total,
                    call.timestep.same_object(&s.time),
                    id(frame.accum) == o.accum,
                    id(frame.counts) == o.counts,
                    id(frame.biases) == o.biases,
                ];
                s.check_callback(&e, "combine-callback", same);
                if s.mode() == "combine-error" {
                    return Err(s.fail("own execute combine failure"));
                }
                if s.mode() == "combine-clear-conditions" {
                    let O::List(v) = &s.conditions else {
                        unreachable!()
                    };
                    v.borrow_mut().clear();
                }
                let last = window_index == total_windows as i64 - 1
                    && o.modality + 1
                        == if s.case["params"]["multi"] == true {
                            2
                        } else {
                            1
                        };
                if last && matches!(s.mode(), "final-counts-error" | "final-and-cleanup-error") {
                    frame.counts.borrow_mut().clear();
                }
                if last && s.mode() == "final-bad-dim" {
                    h.settings.borrow_mut().dim = 7;
                }
            }
            ContextCallbackFrame::ExecuteCleanup { call } => {
                let same = s.same_call(call);
                s.check_callback(&e, "cleanup-callback", same);
                s.cleanup.set(s.cleanup.get() + 1);
                if matches!(s.mode(), "cleanup-error" | "final-and-cleanup-error") {
                    return Err(s.fail("own execute cleanup failure"));
                }
                if s.mode() == "cleanup-mutate" {
                    let root = s.accum.borrow().as_ref().unwrap().upgrade().unwrap();
                    if let Some(row) = root.borrow().first().cloned()
                        && let Some(v) = row.borrow().first()
                    {
                        let _ = v.as_tensor()?.f_add_scalar_(0.5)?;
                    }
                }
            }
        };
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
        check(call.input, &e["input"], s.dev, &s.note);
        check(call.timestep, &e["time"], s.dev, &s.note);
        check_options(call.conditions, &e["conditions"], &s.note);
        if s.mode() == "model-error" {
            return Err(s.fail("own execute model failure"));
        }
        if s.mode() == "model-mutate" {
            let _ = call.input.as_tensor()?.f_add_scalar_(0.25)?;
        }
        let O::List(conds) = call.conditions else {
            panic!("own conditions")
        };
        let mut out = vec![];
        for (ci, cond) in conds.borrow().iter().enumerate() {
            let input = call.input.as_tensor()?;
            let v = if matches!(cond, O::Atom(V::None)) {
                input.f_zeros_like()?
            } else {
                let O::List(entries) = cond else {
                    panic!("own entries")
                };
                let marker = if let Some(entry) = entries.borrow().first() {
                    let O::Atom(V::Int(n)) = entry.get("marker")?.unwrap() else {
                        panic!("own marker")
                    };
                    n
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
            out.push(V::tensor(v));
        }
        Ok(l(out))
    }
}
fn add_row(pairs: &mut Vec<(String, V)>, name: &str, row: &WindowList<V>) {
    for (i, v) in row.borrow().iter().enumerate() {
        pairs.push((format!("{name}.{i}"), v.clone()));
    }
}
fn add_matrix(pairs: &mut Vec<(String, V)>, name: &str, rows: &WindowList<WindowList<V>>) {
    for (i, row) in rows.borrow().iter().enumerate() {
        add_row(pairs, &format!("{name}.{i}"), row);
    }
}
impl ExecutionBindings for Bindings {
    fn warning(&self, message: &str) -> Result<(), Error> {
        self.0
            .warnings
            .borrow_mut()
            .push(json!({"category":"UserWarning","message":message}));
        Ok(())
    }
    fn execution_locals(&self, v: &ExecutionLocals, tag: &str) -> Result<(), Error> {
        let s = &self.0;
        let e = s.next(tag);
        *s.accum.borrow_mut() = Some(Rc::downgrade(&v.accum));
        let mut pairs = vec![];
        add_row(&mut pairs, "state.latents", &v.state.latents);
        for (i, g) in v.state.guide_latents.borrow().iter().enumerate() {
            if let Some(g) = g {
                pairs.push((format!("state.guides.{i}"), g.clone()));
            }
        }
        add_matrix(&mut pairs, "accum", &v.accum);
        add_matrix(&mut pairs, "counts", &v.counts);
        if let Some(results) = &v.results {
            for (i, r) in results.borrow().iter().enumerate() {
                add_matrix(&mut pairs, &format!("results.{i}"), &r.outputs);
            }
        }
        if let Some(r) = &v.result {
            add_matrix(&mut pairs, "result", &r.outputs);
        }
        for (name, row) in [
            ("mod_out", &v.modality_outputs),
            ("result_out", &v.result_out),
            ("finalized", &v.finalized),
        ] {
            if let Some(row) = row {
                add_row(&mut pairs, name, row);
            }
        }
        for (name, x) in [("f", &v.last_final), ("packed", &v.packed)] {
            if let Some(x) = x {
                pairs.push((name.into(), x.clone()));
            }
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
        if tag == "combine-callback" {
            let r = v.result.as_ref().unwrap();
            let window = v.modality_window.as_ref().unwrap();
            let modality = (0..v.state.latents.borrow().len())
                .find(|i| {
                    Rc::ptr_eq(
                        &c::IndexWindow::modality(&r.window, *i as i64).unwrap(),
                        window,
                    )
                })
                .unwrap();
            *s.observed.borrow_mut() = ObservedCombine {
                input: tensor_id(&v.state.latents.borrow()[modality]),
                outputs: id(v.modality_outputs.as_ref().unwrap()),
                conditions: option_list_id(&r.conditions),
                window: id(window),
                accum: id(&v.accum.borrow()[modality]),
                counts: id(&v.counts.borrow()[modality]),
                biases: id(&v.biases.borrow()[modality]),
                index: r.window_index,
                total: v.enumerated.borrow().len(),
                modality,
            };
        }
        Ok(())
    }
}
#[test]
fn whole_context_execution_preserves_schedules_fusion_finally_boundaries_mutations_aliases_and_live_frame_tensors()
 {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-context-execute.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let devices = fixture["devices"].as_array().unwrap();
    assert_eq!(devices, &vec![json!("cpu"), json!("cuda:0")]);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 328 * devices.len());
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = device(p["device"].as_str().unwrap());
        let dtype = kind(p["dtype"].as_str().unwrap());
        rt.native
            .set_default_dtype(kind(p["default"].as_str().unwrap()))
            .unwrap();
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
        let entries = list(
            [vec![2, 1, 1], vec![1, 2, 2]]
                .into_iter()
                .map(|shape| {
                    dict(vec![
                        (
                            "latent_shape",
                            O::Tuple(Rc::new(
                                shape.into_iter().map(|n| O::Atom(V::Int(n))).collect(),
                            )),
                        ),
                        ("pre_filter_count", O::Atom(V::Int(1))),
                    ])
                })
                .collect(),
        );
        let kf = Tensor::zeros([1, 3, 2, 2], (Kind::Int64, dev));
        let times = Tensor::from_slice(&[2i64, 12]).to_device(dev);
        kf.get(0).get(0).select(1, 0).copy_(&times);
        let kf = V::tensor(kf);
        let entry = |marker| {
            let inner = dict(vec![
                (
                    "regular",
                    O::Object(Rc::new(Condition::from_options(
                        CK::Regular,
                        O::Atom(data(&[1, 1, frames, 2, 2], dev, dtype, marker + 3)),
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
                    O::Atom(V::None)
                } else {
                    list(vec![entry(1), entry(2)])
                },
                list(vec![entry(5)]),
            ])
        };
        let transformer = dict(vec![
            ("sample_sigmas", O::Atom(restore(&case["sigmas"], dev))),
            ("own", O::Atom(V::Int(7))),
        ]);
        if p["mode"] == "sample-missing" {
            transformer.pop("sample_sigmas").unwrap();
        }
        let options = dict(vec![("transformer_options", transformer)]);
        let h = Rc::new(Handler::new(
            c::Schedule::from_name(p["schedule"].as_str().unwrap()).unwrap(),
            c::Fuse::from_name(p["fuse"].as_str().unwrap()).unwrap(),
        ));
        h.step.set(7);
        {
            let mut settings = h.settings.borrow_mut();
            settings.dim = 2;
            settings.length = 4;
            settings.overlap = 1;
            settings.stride = 2;
            settings.causal_window_fix = matches!(
                p["schedule"].as_str().unwrap(),
                "looped_uniform" | "batched"
            );
            settings.split_conditions = true;
            settings.condition_retain = l(vec![0]);
            settings.latent_retain = l(vec![0]);
        }
        for key in [
            "execute_start",
            "evaluate_context_windows",
            "combine_context_window_results",
            "execute_cleanup",
        ] {
            h.callbacks.borrow_mut().add(
                key,
                Some("own"),
                Rc::new(O::Object(Rc::new(key.to_string()))),
            );
        }
        let shared = Rc::new(Shared {
            case: case.clone(),
            position: Cell::new(0),
            interrupts: Cell::new(0),
            cleanup: Cell::new(0),
            infos: RefCell::new(vec![]),
            warnings: RefCell::new(vec![]),
            h: Rc::downgrade(&h),
            model: RefCell::new(None),
            input: x.clone(),
            time: time.clone(),
            conditions: conditions.clone(),
            options: options.clone(),
            dev,
            note: format!("context execute {index} {p}"),
            observed: RefCell::new(ObservedCombine::default()),
            accum: RefCell::new(None),
        });
        let model: Rc<dyn ContextModel> = Rc::new(Model);
        *shared.model.borrow_mut() = Some(Rc::downgrade(&model));
        let outcome = Handler::execute(
            &rt,
            &Bindings(shared.clone()),
            ContextCall {
                handler: &h,
                model: &model,
                input: &x,
                conditions: &conditions,
                timestep: &time,
                options: &options,
            },
        );
        match outcome {
            Ok(v) => {
                assert_eq!(case["ok"], true, "{}", shared.note);
                check(
                    &V::List(v.borrow().clone()),
                    &case["outputs"],
                    dev,
                    &shared.note,
                );
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{} {e}", shared.note);
                reference_error(e, case);
            }
        }
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
        assert_eq!(
            json!(*shared.warnings.borrow()),
            case["warnings"],
            "{}",
            shared.note
        );
        let same_model = match &*h.model.borrow() {
            ResizeModelSlot::Bound(v) => Rc::ptr_eq(v, &(model.clone() as Rc<dyn ResizeModel>)),
            _ => false,
        };
        let O::List(conds) = &conditions else {
            unreachable!()
        };
        assert_eq!(
            json!({"step":h.step.get(),"dim":h.settings.borrow().dim,"fuse":h.settings.borrow().fuse.name(),"conditions_count":conds.borrow().len(),"same_model":same_model,"model_none":matches!(*h.model.borrow(),ResizeModelSlot::None),"cleanup":shared.cleanup.get()}),
            case["post"],
            "{}",
            shared.note
        );
    }
    rt.native.set_default_dtype(Kind::Float).unwrap();
}
