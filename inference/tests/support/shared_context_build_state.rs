use super::{
    binding_tests::{check, reference_error, restore},
    device, kind, kitchen,
    shared_context_state_tests::{data, l},
};
use imagine_inference::{
    conditioning::{Condition, Kind as CK},
    context_windows::{
        self as c, BuiltinConditionAttributes, Handler, LatentShape, LatentShapes,
        WindowBuildLocals, WindowStateModel,
    },
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options as O,
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};
use tch::Device;
fn list(v: Vec<O>) -> O {
    O::List(l(v))
}
fn dict(v: Vec<(&str, O)>) -> O {
    O::dictionary(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn constant(v: O) -> O {
    O::Object(Rc::new(Condition::from_options(CK::Constant, v)))
}
fn none() -> O {
    O::Atom(V::None)
}
fn same(a: &O, b: &O) -> bool {
    a.same_object(b) || matches!((a, b), (O::Atom(V::None), O::Atom(V::None)))
}
fn slot(value: Option<&O>, payload: &O) -> bool {
    value.map_or_else(|| matches!(payload, O::Atom(V::None)), |v| same(v, payload))
}
fn ownership(values: &[V]) -> Value {
    json!({"same_objects":values.iter().map(|a|values.iter().map(|b|a.same_object(b)).collect::<Vec<_>>()).collect::<Vec<_>>(),"same_storages":values.iter().map(|a|values.iter().map(|b|native::storage_info(&a.as_tensor().unwrap()).unwrap().0==native::storage_info(&b.as_tensor().unwrap()).unwrap().0).collect::<Vec<_>>()).collect::<Vec<_>>()})
}
struct Model {
    events: Vec<Value>,
    position: Cell<usize>,
    input: V,
    shapes: O,
    entries: O,
    keyframes: O,
    h: Rc<Handler>,
    mode: String,
    note: String,
    dev: Device,
}
impl WindowStateModel for Model {
    fn source_locals(&self, locals: &WindowBuildLocals<'_>) -> Result<(), Error> {
        let expected = &self.events[self.position.get()];
        self.position.set(self.position.get() + 1);
        let mut tensors = vec![("x_in".to_string(), self.input.clone())];
        for (name, values) in [
            ("unpacked_latents", locals.unpacked.to_vec()),
            ("unpacked_latents_list", locals.latents.borrow().clone()),
            (
                "guide_latents_list",
                locals
                    .guide_latents
                    .borrow()
                    .iter()
                    .filter_map(Clone::clone)
                    .collect(),
            ),
            (
                "keyframe_idxs_list",
                locals
                    .keyframe_idxs
                    .borrow()
                    .iter()
                    .filter_map(Clone::clone)
                    .collect(),
            ),
        ] {
            // Only primary guide/keyframe slots can be populated by this body.
            for (i, value) in values.into_iter().enumerate() {
                tensors.push((format!("{name}.{i}"), value));
            }
        }
        if let O::Atom(v @ V::Tensor(_)) = locals.extracted_keyframes {
            tensors.push(("extracted_keyframe_idxs".into(), v.clone()));
        }
        if let Some(v) = locals.primary {
            tensors.push(("x".into(), v.clone()));
        }
        assert_eq!(
            json!(tensors.iter().map(|(k, _)| k).collect::<Vec<_>>()),
            expected["tensor_names"],
            "{}",
            self.note
        );
        assert_eq!(
            tensors.len(),
            expected["tensors"].as_object().unwrap().len(),
            "{}",
            self.note
        );
        for (k, v) in &tensors {
            check(
                v,
                &expected["tensors"][k],
                self.dev,
                &format!("{} local {k}", self.note),
            );
        }
        let owners = ownership(&tensors.into_iter().map(|(_, v)| v).collect::<Vec<_>>());
        assert_eq!(
            owners["same_objects"], expected["same_objects"],
            "{}",
            self.note
        );
        assert_eq!(
            owners["same_storages"], expected["same_storages"],
            "{}",
            self.note
        );
        assert_eq!(
            same(locals.source_shapes, &self.shapes),
            expected["same_shapes"],
            "{}",
            self.note
        );
        assert_eq!(
            same(locals.extracted_guides, &self.entries),
            expected["same_guides"],
            "{}",
            self.note
        );
        assert_eq!(
            same(locals.extracted_keyframes, &self.keyframes),
            expected["same_keyframes"],
            "{}",
            self.note
        );
        assert_eq!(
            json!(
                locals
                    .guide_entries
                    .borrow()
                    .iter()
                    .map(|v| slot(v.as_ref(), &self.entries))
                    .collect::<Vec<_>>()
            ),
            expected["same_entry_slots"],
            "{}",
            self.note
        );
        assert_eq!(
            json!(locals.is_multimodal),
            expected["is_multimodal"],
            "{}",
            self.note
        );
        Ok(())
    }
    fn temporal_downscale_ratio(&self) -> Result<i64, Error> {
        if self.mode == "mutate" {
            self.h.settings.borrow_mut().dim = 1;
            if let O::List(v) = &self.shapes {
                v.borrow_mut().push(O::Tuple(Rc::new(
                    vec![1, 1, 9, 1, 1]
                        .into_iter()
                        .map(|n| O::Atom(V::Int(n)))
                        .collect(),
                )));
            }
            if let O::List(v) = &self.entries {
                v.borrow_mut().push(dict(vec![
                    (
                        "latent_shape",
                        O::Tuple(Rc::new(
                            vec![1, 1, 1]
                                .into_iter()
                                .map(|n| O::Atom(V::Int(n)))
                                .collect(),
                        )),
                    ),
                    ("pre_filter_count", O::Atom(V::Int(1))),
                ]));
            }
        }
        if self.mode == "error" {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own temporal ratio failure".into(),
            });
        }
        Ok(2)
    }
}
#[test]
fn whole_context_state_build_preserves_metadata_read_order_live_shapes_guide_views_and_model_boundary_owners()
 {
    let _rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-build-state.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1216);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let note = format!("context state build {index} {p}");
        let dev = device(p["device"].as_str().unwrap());
        let dtype = kind(p["dtype"].as_str().unwrap());
        let mode = p["mode"].as_str().unwrap();
        let x = restore(&case["input"], dev);
        let mut frames = vec![12];
        if ["two", "three"].contains(&mode) {
            frames.push(6);
        }
        if mode == "three" {
            frames.push(18);
        }
        let sizes = frames
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                let sizes = if i == 0 {
                    vec![1, 1, n, 1, 2]
                } else {
                    vec![1, 1, n, 1, 1]
                };
                let values = sizes.iter().map(|n| O::Atom(V::Int(*n))).collect();
                match p["shape_kind"].as_str().unwrap() {
                    "size" => O::Object(Rc::new(LatentShape::Size(Rc::new(sizes)))),
                    "tuple" => O::Tuple(Rc::new(values)),
                    _ => list(values),
                }
            })
            .collect();
        let shapes = if mode == "single" {
            none()
        } else if p["root_kind"] == "tuple" {
            O::Tuple(Rc::new(sizes))
        } else {
            list(sizes)
        };
        let guides = p["guides"].as_str().unwrap();
        let entries = match guides {
            "none" => none(),
            "empty" => list(vec![]),
            _ => list(vec![dict(vec![
                (
                    "latent_shape",
                    O::Tuple(Rc::new(
                        vec![guides.parse().unwrap(), 1, 1]
                            .into_iter()
                            .map(|n| O::Atom(V::Int(n)))
                            .collect(),
                    )),
                ),
                ("pre_filter_count", O::Atom(V::Int(1))),
            ])]),
        };
        let keyframes = if p["has_keyframes"] == true {
            O::Atom(data(&[1, 2, 1, 2], dev, dtype, 7))
        } else {
            none()
        };
        let model_conds = dict(vec![]);
        if mode != "single" {
            model_conds
                .set("latent_shapes", constant(shapes.clone()))
                .unwrap();
        }
        if guides != "none" {
            model_conds
                .set("guide_attention_entries", constant(entries.clone()))
                .unwrap();
        }
        if p["has_keyframes"] == true {
            model_conds
                .set("keyframe_idxs", constant(keyframes.clone()))
                .unwrap();
        }
        let conditions = list(vec![
            none(),
            list(vec![dict(vec![]), dict(vec![("model_conds", model_conds)])]),
        ]);
        let h = Rc::new(Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat));
        h.settings.borrow_mut().dim = 2;
        let model = Model {
            events: case["events"].as_array().unwrap().clone(),
            position: Cell::new(0),
            input: x.clone(),
            shapes: shapes.clone(),
            entries: entries.clone(),
            keyframes: keyframes.clone(),
            h: h.clone(),
            mode: p["model_mode"].as_str().unwrap().into(),
            note: note.clone(),
            dev,
        };
        let result = h.build_window_state(&x, &conditions, &BuiltinConditionAttributes, &model);
        assert_eq!(model.position.get(), model.events.len(), "{note}");
        assert_eq!(json!(h.settings.borrow().dim), case["h_dim"], "{note}");
        if matches!(shapes, O::Atom(V::None)) {
            assert!(case["shapes_post"].is_null(), "{note}");
        } else {
            let current = LatentShapes::from_options(&shapes).unwrap();
            assert_eq!(
                json!(
                    current
                        .entries()
                        .unwrap()
                        .iter()
                        .map(|v| v.values().unwrap())
                        .collect::<Vec<_>>()
                ),
                case["shapes_post"],
                "{note}"
            );
        }
        match result {
            Ok(state) => {
                assert!(!case["error"].is_string(), "{note}");
                let optional = |values: &c::WindowList<Option<V>>| {
                    V::List(
                        values
                            .borrow()
                            .iter()
                            .map(|v| v.clone().unwrap_or(V::None))
                            .collect(),
                    )
                };
                check(
                    &V::Dict(vec![
                        ("latents".into(), V::List(state.latents.borrow().clone())),
                        ("guide_latents".into(), optional(&state.guide_latents)),
                        ("keyframes".into(), optional(&state.keyframe_idxs)),
                        ("dim".into(), V::Int(state.dim)),
                        ("multi".into(), V::Bool(state.is_multimodal)),
                        ("ratio".into(), V::Int(state.temporal_downscale_ratio)),
                    ]),
                    &case["output"],
                    dev,
                    &note,
                );
                assert_eq!(
                    state.latent_shapes.as_ref().map_or_else(
                        || matches!(shapes, O::Atom(V::None)),
                        |v| v.same_object(&LatentShapes::from_options(&shapes).unwrap())
                    ),
                    case["owners"]["same_shapes"],
                    "{note}"
                );
                assert_eq!(
                    json!(
                        state
                            .guide_entries
                            .borrow()
                            .iter()
                            .map(|v| slot(v.as_ref(), &entries))
                            .collect::<Vec<_>>()
                    ),
                    case["owners"]["same_entries"],
                    "{note}"
                );
                let mut values = vec![x.clone()];
                values.extend(state.latents.borrow().clone());
                values.extend(state.guide_latents.borrow().iter().filter_map(Clone::clone));
                values.extend(state.keyframe_idxs.borrow().iter().filter_map(Clone::clone));
                let owners = ownership(&values);
                assert_eq!(
                    owners["same_objects"], case["owners"]["same_objects"],
                    "{note}"
                );
                assert_eq!(
                    owners["same_storages"], case["owners"]["same_storages"],
                    "{note}"
                );
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: {e}");
                reference_error(e, case);
            }
        }
    }
}
