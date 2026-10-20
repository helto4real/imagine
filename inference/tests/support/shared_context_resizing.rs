use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
    shared_context_state_tests::{data, l},
};
use imagine_inference::{
    conditioning::{Condition, Kind as CK},
    context_windows::{
        self as c, Handler, IndexWindow, ResizeLocals, ResizeModel, ResizeModelSlot,
        ResizingBindings, WindowHandle, WindowList,
    },
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
    patching::{extensions::Original, options::Options as O},
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use tch::{Device, Tensor};

fn atom(v: V) -> O {
    O::Atom(v)
}
fn text(v: &str) -> O {
    atom(V::Text(v.into()))
}
fn dict(v: Vec<(&str, O)>) -> O {
    O::dictionary(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn list(v: Vec<O>) -> O {
    O::List(l(v))
}
fn cond(kind: CK, value: O) -> O {
    O::Object(Rc::new(Condition::from_options(kind, value)))
}
struct Control;
const NAMES: [&str; 12] = [
    "resized_cond",
    "actual_cond",
    "resized_actual_cond",
    "cond_item",
    "actual_cond_item",
    "new_cond_item",
    "cond_value",
    "callback",
    "result",
    "audio_cond",
    "vace_cond",
    "sliced_vace",
];

/// Canonical ordered object/storage graph, independently walked on each side.
/// Expected tensor bytes are checked by the common exact comparator, not copied
/// into the native result. These temporary observer owners die before execution resumes.
struct Graph<'a> {
    nodes: Vec<O>,
    storages: Vec<usize>,
    labels: &'a [(O, &'static str)],
    note: &'a str,
}
impl Graph<'_> {
    fn check(&mut self, value: &O, expected: &Value) {
        if let O::Atom(v @ (V::None | V::Bool(_) | V::Int(_) | V::Float(_) | V::Text(_))) = value {
            assert_eq!(expected["type"], "scalar", "{}", self.note);
            check(v, expected, Device::Cpu, self.note);
            return;
        }
        if let Some(index) = self.nodes.iter().position(|v| v.same_object(value)) {
            assert_eq!(*expected, json!({"ref":index}), "{}", self.note);
            return;
        }
        let index = self.nodes.len();
        self.nodes.push(value.clone());
        assert_eq!(expected["id"], index, "{}", self.note);
        match value {
            O::Atom(v @ V::Tensor(_)) => {
                assert_eq!(expected["type"], "tensor", "{}", self.note);
                let t = v.as_tensor().unwrap();
                check(v, &expected["value"], t.device(), self.note);
                let storage = native::storage_info(&t).unwrap().0;
                let index = self
                    .storages
                    .iter()
                    .position(|v| *v == storage)
                    .unwrap_or_else(|| {
                        self.storages.push(storage);
                        self.storages.len() - 1
                    });
                assert_eq!(expected["storage"], index, "{}", self.note);
            }
            O::Dict(v) => {
                assert_eq!(expected["type"], "dict", "{}", self.note);
                let v = v.borrow();
                let entries = expected["items"].as_array().unwrap();
                assert_eq!(v.len(), entries.len(), "{}", self.note);
                for ((k, v), e) in v.iter().zip(entries) {
                    assert_eq!(k, e[0].as_str().unwrap(), "{}", self.note);
                    self.check(v, &e[1]);
                }
            }
            O::List(v) => {
                assert_eq!(expected["type"], "list", "{}", self.note);
                self.sequence(&v.borrow(), expected);
            }
            O::Tuple(v) => {
                assert_eq!(expected["type"], "tuple", "{}", self.note);
                self.sequence(v, expected);
            }
            O::Object(v) => {
                if let Some(c) = v.downcast_ref::<Condition>() {
                    assert_eq!(expected["type"], "condition", "{}", self.note);
                    let name = match c.kind {
                        CK::Regular => "CONDRegular",
                        CK::NoiseShape => "CONDNoiseShape",
                        CK::CrossAttention => "CONDCrossAttn",
                        CK::Constant => "CONDConstant",
                        CK::List => "CONDList",
                    };
                    assert_eq!(expected["class_name"], name, "{}", self.note);
                    self.check(&c.source_value(), &expected["value"]);
                } else if v.is::<Control>() {
                    assert_eq!(expected["type"], "control", "{}", self.note);
                } else {
                    assert_eq!(expected["type"], "callback", "{}", self.note);
                    let label = self
                        .labels
                        .iter()
                        .find(|(v, _)| v.same_object(value))
                        .unwrap()
                        .1;
                    assert_eq!(expected["label"], label, "{}", self.note);
                }
            }
            _ => panic!("unexpected graph value: {}", self.note),
        }
    }
    fn sequence(&mut self, values: &[O], expected: &Value) {
        let items = expected["items"].as_array().unwrap();
        assert_eq!(values.len(), items.len(), "{}", self.note);
        for (v, e) in values.iter().zip(items) {
            self.check(v, e);
        }
    }
}
fn graph(value: &O, expected: &Value, labels: &[(O, &'static str)], note: &str) {
    Graph {
        nodes: vec![],
        storages: vec![],
        labels,
        note,
    }
    .check(value, expected);
}
struct Bindings {
    expected: Vec<Value>,
    position: Cell<usize>,
    labels: RefCell<Vec<(O, &'static str)>>,
    infos: RefCell<Vec<String>>,
    note: String,
}
impl ResizingBindings for Bindings {
    fn source_locals(&self, locals: &ResizeLocals, event: &str) -> Result<(), Error> {
        let i = self.position.get();
        let expected = &self.expected[i];
        self.position.set(i + 1);
        assert_eq!(expected["kind"], event, "{} event {i}", self.note);
        let all: Vec<_> = locals.values().collect();
        let values = O::dictionary(
            NAMES
                .iter()
                .filter_map(|name| {
                    all.iter()
                        .find(|(key, _)| key == name)
                        .map(|(_, v)| (name.to_string(), (*v).clone()))
                })
                .collect(),
        );
        graph(
            &values,
            &expected["locals"],
            &self.labels.borrow(),
            &format!("{} {event} {i}", self.note),
        );
        Ok(())
    }
    fn info(&self, message: &str) -> Result<(), Error> {
        self.infos.borrow_mut().push(message.into());
        Ok(())
    }
    fn prepare_control_objects(&self, value: &O, _: Option<Device>) -> Result<O, Error> {
        if let O::Object(v) = value
            && v.is::<Control>()
        {
            return Ok(value.clone());
        }
        let typename = if matches!(value, O::Atom(V::None)) {
            "NoneType"
        } else {
            panic!("unexpected synthetic control")
        };
        Err(Error::ReferenceException {
            class: "AttributeError",
            message: format!("'{typename}' object has no attribute 'previous_controlnet'"),
        })
    }
}
struct Model {
    handler: Weak<Handler>,
    mode: String,
    bindings: Rc<Bindings>,
}
impl ResizeModel for Model {
    fn resize_cond_for_context_window(
        &self,
        _: &str,
        value: &O,
        _: &WindowHandle,
        _: &V,
        _: Option<Device>,
        retain: WindowList<i64>,
    ) -> Result<O, Error> {
        let h = self.handler.upgrade().unwrap();
        assert_eq!(
            self.bindings.expected[self.bindings.position.get() - 1]["same_retain"],
            Rc::ptr_eq(&retain, &h.settings.borrow().condition_retain)
        );
        match self.mode.as_str() {
            "model-handle" => return Ok(value.clone()),
            "model-clear" => *h.model.borrow_mut() = ResizeModelSlot::None,
            "model-retain" => retain.borrow_mut().push(1),
            _ => {}
        }
        Ok(atom(V::None))
    }
}
#[test]
fn whole_context_resizing_preserves_callbacks_models_mutations_device_moves_object_graphs_and_source_locals()
 {
    let _rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-context-resizing.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 832);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = device(p["device"].as_str().unwrap());
        let dtype = kind(&format!("torch.{}", p["dtype"].as_str().unwrap()));
        let dim = p["dim"].as_i64().unwrap();
        let profile = p["profile"].as_str().unwrap();
        let mode = p["mode"].as_str().unwrap();
        let split = p["split"].as_bool().unwrap();
        let target = p["target"].as_str().map(device);
        let mut shape = vec![1; 5];
        shape[dim as usize] = if profile == "batch1" { 1 } else { 8 };
        let x = data(&shape, dev, dtype, 0);
        let w = IndexWindow::new(
            l(if profile == "batch1" {
                vec![0, 2, 4, 6]
            } else {
                vec![1, 3, 4, 6]
            }),
            dim,
            shape[dim as usize],
            None,
            1,
        )
        .unwrap();
        if p["empty_window"] == true {
            w.borrow().indices.borrow_mut().clear();
        }
        if p["anchor"] == true {
            w.borrow_mut().causal_anchor = Some(0);
        }
        let h = Rc::new(Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat));
        {
            let mut s = h.settings.borrow_mut();
            s.length = 4;
            s.overlap = 1;
            s.dim = dim;
            s.condition_retain = l(vec![0]);
            s.split_conditions = split;
        }
        let inner = dict(vec![
            ("raw", atom(data(&shape, dev, dtype, 2))),
            (
                "regular",
                cond(CK::Regular, atom(data(&shape, dev, dtype, 3))),
            ),
            (
                "cross",
                cond(CK::CrossAttention, atom(data(&[1, 3, 1], dev, dtype, 4))),
            ),
            (
                "constant",
                cond(CK::Constant, list(vec![text("own"), atom(V::Int(3))])),
            ),
            ("num_video_frames", cond(CK::Constant, atom(V::Int(8)))),
            (
                "list_cond",
                cond(CK::List, list(vec![atom(data(&[1, 2], dev, dtype, 5))])),
            ),
        ]);
        if profile == "audio-vace" {
            inner
                .set(
                    "audio_embed",
                    cond(CK::Regular, atom(data(&[1, 8, 1], dev, dtype, 6))),
                )
                .unwrap();
            inner
                .set(
                    "vace_context",
                    cond(CK::NoiseShape, atom(data(&[1, 1, 1, 8, 1], dev, dtype, 7))),
                )
                .unwrap();
        }
        if profile == "low-rank" {
            inner.set("raw", atom(data(&[8], dev, dtype, 0))).unwrap();
        }
        let inner = if profile == "scalars" {
            dict(vec![
                ("num_video_frames", cond(CK::Constant, atom(V::Int(8)))),
                ("plain", atom(V::Int(3))),
                ("none", atom(V::None)),
                ("false", atom(V::Bool(false))),
            ])
        } else {
            inner
        };
        let first = dict(vec![
            ("top", atom(data(&shape, dev, dtype, 1))),
            ("control", O::Object(Rc::new(Control))),
            ("model_conds", inner.clone()),
            ("marker", text("own")),
        ]);
        if profile == "control-none" {
            first.set("control", atom(V::None)).unwrap();
        }
        if profile == "control-tensor" {
            first
                .set("control", atom(data(&shape, dev, dtype, 9)))
                .unwrap();
        }
        if profile == "shared-cond" {
            inner
                .set("regular_again", inner.get("regular").unwrap().unwrap())
                .unwrap();
        }
        let mut input = vec![first.clone()];
        if split || profile == "shared-map" {
            input.push(dict(vec![
                (
                    "model_conds",
                    if profile == "shared-map" {
                        inner.clone()
                    } else {
                        dict(vec![("marker", cond(CK::Constant, text("second")))])
                    },
                ),
                ("marker", text("second")),
            ]));
        }
        let input = match profile {
            "none" => atom(V::None),
            "empty" => list(vec![]),
            "tuple-input" => O::Tuple(Rc::new(input)),
            _ => list(input),
        };
        let captured = list(vec![]);
        let bindings = Rc::new(Bindings {
            expected: case["events"].as_array().unwrap().clone(),
            position: Cell::new(0),
            labels: RefCell::new(vec![]),
            infos: RefCell::new(vec![]),
            note: format!("context resizing {index} {p}"),
        });
        let mut callbacks = vec![];
        for second in [false, true] {
            let handler = Rc::downgrade(&h);
            let mode = mode.to_string();
            let shape = shape.clone();
            let captured = captured.clone();
            let input = input.clone();
            let first = first.clone();
            let called = Cell::new(false);
            let callback: Original<O> = Rc::new(move |args, _| {
                let O::Atom(V::Text(key)) = &args[0] else {
                    panic!("key")
                };
                let value = &args[1];
                let mapping = &args[5];
                let O::List(captured) = &captured else {
                    unreachable!()
                };
                captured.borrow_mut().push(mapping.clone());
                if second {
                    return Ok(value.clone());
                }
                if mode == "handle-false" {
                    return Ok(atom(V::Bool(false)));
                }
                if mode == "raise" {
                    return Err(Error::ReferenceException {
                        class: "ValueError",
                        message: "own resize callback failure".into(),
                    });
                }
                if mode == "mutate-entry" {
                    mapping.set(
                        key,
                        atom(V::tensor(Tensor::full(&shape[..], 9., (dtype, dev)))),
                    )?;
                }
                if mode == "mutate-cond"
                    && let O::Object(v) = value
                    && let Some(c) = v.downcast_ref::<Condition>()
                    && let O::Atom(v @ V::Tensor(_)) = c.source_value()
                {
                    c.set_value(atom(V::tensor(v.as_tensor()?.f_add_scalar(1.25)?)));
                }
                if !called.replace(true) {
                    match mode.as_str() {
                        "append-input" => {
                            if let O::List(input) = &input {
                                input
                                    .borrow_mut()
                                    .push(dict(vec![("marker", text("added"))]));
                            }
                        }
                        "add-inner" => mapping.set("late", cond(CK::Constant, text("late")))?,
                        "reinsert-inner" => {
                            mapping.pop(key)?;
                            mapping.set(key, value.clone())?;
                        }
                        "outer-add" => first.set("late_top", text("late"))?,
                        "mutate-dim" => handler.upgrade().unwrap().settings.borrow_mut().dim = 1,
                        _ => {}
                    }
                }
                Ok(atom(V::None))
            });
            let callback = O::Object(Rc::new(callback));
            bindings
                .labels
                .borrow_mut()
                .push((callback.clone(), if second { "second" } else { "first" }));
            callbacks.push(callback);
        }
        if [
            "observe",
            "handle-false",
            "handle-second",
            "mutate-entry",
            "mutate-cond",
            "append-input",
            "add-inner",
            "reinsert-inner",
            "outer-add",
            "mutate-dim",
            "raise",
        ]
        .contains(&mode)
        {
            h.callbacks.borrow_mut().add(
                "resize_cond_item",
                Some("first"),
                Rc::new(callbacks[0].clone()),
            );
            if mode == "handle-second" {
                h.callbacks.borrow_mut().add(
                    "resize_cond_item",
                    Some("second"),
                    Rc::new(callbacks[1].clone()),
                );
            }
        } else if mode == "bad-callback" {
            h.callbacks
                .borrow_mut()
                .add("resize_cond_item", Some("bad"), Rc::new(atom(V::Int(3))));
        }
        *h.model.borrow_mut() = if p["missing_with_handled"] == true || mode == "model-missing" {
            ResizeModelSlot::Missing
        } else if mode.starts_with("model-") {
            ResizeModelSlot::Bound(Rc::new(Model {
                handler: Rc::downgrade(&h),
                mode: mode.into(),
                bindings: bindings.clone(),
            }))
        } else {
            ResizeModelSlot::None
        };
        graph(
            &input,
            &case["initial"],
            &bindings.labels.borrow(),
            &bindings.note,
        );
        let result = h.get_resized_cond(bindings.as_ref(), &input, &x, &w, target);
        let output = match result {
            Ok(v) => {
                assert_eq!(case["ok"], true, "{}", bindings.note);
                v
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{} {e}", bindings.note);
                reference_error(e, case);
                atom(V::None)
            }
        };
        assert_eq!(
            bindings.position.get(),
            bindings.expected.len(),
            "{}",
            bindings.note
        );
        assert_eq!(
            json!(*bindings.infos.borrow()),
            case["infos"],
            "{}",
            bindings.note
        );
        graph(
            &dict(vec![
                ("input", input),
                ("output", output),
                ("captured", captured),
            ]),
            &case["post"],
            &bindings.labels.borrow(),
            &bindings.note,
        );
        assert_eq!(
            json!(h.settings.borrow().dim),
            case["dim"],
            "{}",
            bindings.note
        );
        assert_eq!(
            json!(*h.settings.borrow().condition_retain.borrow()),
            case["retain"],
            "{}",
            bindings.note
        );
        assert_eq!(
            match *h.model.borrow() {
                ResizeModelSlot::Missing => "missing",
                ResizeModelSlot::None => "none",
                ResizeModelSlot::Bound(_) => "bound",
            },
            case["model"].as_str().unwrap(),
            "{}",
            bindings.note
        );
    }
}
