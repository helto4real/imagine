use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
};
use imagine_inference::{
    conditioning::{Condition, Kind as CondKind},
    context_windows::{
        IndexWindow, LatentShape, LatentShapes, ModalityIndices, ModalityMapper, WindowHandle,
        WindowList, WindowingState,
    },
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Kind, Tensor};
pub(super) fn l<T>(v: Vec<T>) -> WindowList<T> {
    Rc::new(RefCell::new(v))
}
fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn list(v: Vec<i64>) -> V {
    V::List(v.into_iter().map(V::Int).collect())
}
pub(super) fn snapshot(window: &WindowHandle) -> V {
    let w = window.borrow();
    let guides = V::Tuple(vec![
        list(w.guide_frames_indices.borrow().clone()),
        V::List(
            w.guide_overlap_info
                .borrow()
                .iter()
                .map(|(a, b)| V::Tuple(vec![V::Int(*a), V::Int(*b)]))
                .collect(),
        ),
        list(w.guide_kf_local_positions.borrow().clone()),
        list(w.guide_downscale_factors.borrow().clone()),
    ]);
    V::Dict(vec![
        ("indices".into(), list(w.indices.borrow().clone())),
        ("length".into(), V::Int(w.context_length as i64)),
        ("overlap".into(), V::Int(w.overlap)),
        ("dim".into(), V::Int(w.dim)),
        ("frames".into(), V::Int(w.total_frames)),
        ("center".into(), V::Float(w.center_ratio)),
        ("guides".into(), guides),
    ])
}
pub(super) fn data(shape: &[i64], dev: Device, dtype: Kind, offset: i64) -> V {
    V::tensor(
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
            .to_kind(dtype),
    )
}
struct Mapper {
    action: String,
    original: LatentShapes,
    window: WindowHandle,
    mapped: ModalityIndices,
    calls: Vec<Value>,
}
impl ModalityMapper for Mapper {
    fn class_name(&self) -> &str {
        "NativeMapper"
    }
    fn map_context_window_to_modalities(
        &mut self,
        indices: WindowList<i64>,
        shapes: LatentShapes,
        dim: i64,
    ) -> Result<ModalityIndices, Error> {
        let entries = shapes.entries()?;
        self.calls.push(json!({"indices":indices.borrow().clone(),"dim":dim,"shapes":entries.iter().map(LatentShape::values).collect::<Result<Vec<_>,_>>()?,"types":entries.iter().map(LatentShape::class_name).collect::<Vec<_>>(),"same_shapes":shapes.same_object(&self.original),"same_shape_objects":entries.iter().zip(self.original.entries()?.iter()).map(|(a,b)|a.same_object(b)).collect::<Vec<_>>(),"same_indices":Rc::ptr_eq(&indices,&self.window.borrow().indices)}));
        if self.action == "mutate" {
            indices.borrow_mut().push(1);
        }
        if let Some(class) = match self.action.as_str() {
            "attribute-error" => Some("AttributeError"),
            "value-error" => Some("ValueError"),
            _ => None,
        } {
            return Err(Error::ReferenceException {
                class,
                message: if class == "AttributeError" {
                    "own mapper attribute failure"
                } else {
                    "own mapper value failure"
                }
                .into(),
            });
        }
        Ok(self.mapped.clone())
    }
}
#[test]
fn original_multimodal_window_state_preserves_mapping_aliases_guide_injection_stripping_and_partial_errors()
 {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-context-state.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1128);
    let histories = cases.iter().map(|c| (c, false)).chain(
        cases
            .iter()
            .filter(|c| c["name"] == "prepare")
            .map(|c| (c, true)),
    );
    for (i, (case, source_backed)) in histories.enumerate() {
        let p = &case["params"];
        let dev = p["device"].as_str().map(device).unwrap_or(Device::Cpu);
        let note = format!(
            "window state {i} source_backed={source_backed} {} {p}",
            case["name"]
        );
        if case["name"] == "patch-shapes" {
            let state = WindowingState {
                latents: l(vec![]),
                guide_latents: l(vec![]),
                guide_entries: l(vec![]),
                keyframe_idxs: l(vec![]),
                latent_shapes: None,
                dim: 0,
                is_multimodal: p["multi"].as_bool().unwrap(),
                temporal_downscale_ratio: 1,
            };
            let shapes = Options::from_value(V::List(vec![
                list(vec![1, 1, 3, 2, 2]),
                list(vec![1, 1, 2, 2, 2]),
            ]));
            let old = Rc::new(Condition::new(
                CondKind::Regular,
                V::Text("old-latent-shapes".into()),
            ));
            let shared = Options::from_value(V::Dict(vec![]));
            shared
                .set("latent_shapes", Options::Object(old.clone()))
                .unwrap();
            let first = Options::from_value(V::Dict(vec![]));
            first.set("model_conds", shared.clone()).unwrap();
            let second_map = if p["scenario"] == "shared-map" {
                shared
            } else {
                let m = Options::from_value(V::Dict(vec![]));
                m.set("latent_shapes", Options::Object(old.clone()))
                    .unwrap();
                m
            };
            let second = Options::from_value(V::Dict(vec![]));
            second.set("model_conds", second_map).unwrap();
            let row = |v: Vec<Options>| Options::List(l(v));
            let rows = match p["scenario"].as_str().unwrap() {
                "empty" => vec![],
                "missing" => vec![row(vec![Options::from_value(V::Dict(vec![]))])],
                "shared-entry" => vec![row(vec![first.clone(), first])],
                "skip-none" => vec![Options::Atom(V::None), row(vec![first, second])],
                _ => vec![row(vec![first, second])],
            };
            let conds = row(rows);
            let entries_of = |value: &Options| {
                let Options::List(rows) = value else {
                    unreachable!()
                };
                rows.borrow()
                    .iter()
                    .flat_map(|row| match row {
                        Options::List(entries) => entries.borrow().clone(),
                        _ => vec![],
                    })
                    .collect::<Vec<_>>()
            };
            let before = entries_of(&conds);
            state.patch_latent_shapes(&conds, &shapes).unwrap();
            let after = entries_of(&conds);
            assert_eq!(before.len(), after.len());
            assert_eq!(
                before.iter().zip(&after).all(|(a, b)| a.same_object(b)),
                case["same_entries"].as_bool().unwrap(),
                "{note}"
            );
            let Options::List(rows) = conds else {
                unreachable!()
            };
            let mut values = vec![];
            let mut objects = vec![];
            for row in rows.borrow().iter() {
                if let Options::List(entries) = row {
                    for entry in entries.borrow().iter() {
                        if let Some(map) = entry.get("model_conds").unwrap()
                            && let Some(Options::Object(object)) = map.get("latent_shapes").unwrap()
                        {
                            let cond = object.downcast_ref::<Condition>().unwrap();
                            values.push(V::Dict(vec![
                                (
                                    "class_name".into(),
                                    V::Text(
                                        if cond.kind == CondKind::Constant {
                                            "CONDConstant"
                                        } else {
                                            "CONDRegular"
                                        }
                                        .into(),
                                    ),
                                ),
                                ("same_old".into(), V::Bool(std::ptr::eq(cond, old.as_ref()))),
                                (
                                    "same_shapes".into(),
                                    V::Bool(cond.source_value().same_object(&shapes)),
                                ),
                            ]));
                            objects.push(object);
                        }
                    }
                }
            }
            let aliases = objects
                .iter()
                .map(|a| V::List(objects.iter().map(|b| V::Bool(Rc::ptr_eq(a, b))).collect()))
                .collect();
            check(
                &V::Tuple(vec![V::List(values), V::List(aliases)]),
                &case["output"],
                dev,
                &note,
            );
            check(&old.value().unwrap(), &case["old"], dev, &note);
            continue;
        }
        let (out, window) = if case["name"] == "prepare" {
            let primary = p["primary"].as_i64().unwrap();
            let working = p["working"].as_i64().unwrap();
            let shapes = LatentShapes::new(
                [primary, 4, 12]
                    .into_iter()
                    .map(|n| {
                        let v = vec![1, 1, n, 2, 2];
                        match p["shape_kind"].as_str().unwrap() {
                            "size" => LatentShape::Size(Rc::new(v)),
                            "tuple" => LatentShape::Tuple(Rc::new(v)),
                            _ => LatentShape::List(l(v)),
                        }
                    })
                    .collect(),
            );
            let shapes = if source_backed {
                let entries = shapes
                    .entries()
                    .unwrap()
                    .into_iter()
                    .map(|v| {
                        let values = v.values().unwrap();
                        match v {
                            LatentShape::Size(_) => Options::Object(Rc::new(v)),
                            LatentShape::Tuple(_) => Options::Tuple(Rc::new(
                                values
                                    .into_iter()
                                    .map(|v| Options::Atom(V::Int(v)))
                                    .collect(),
                            )),
                            _ => Options::List(l(values
                                .into_iter()
                                .map(|v| Options::Atom(V::Int(v)))
                                .collect())),
                        }
                    })
                    .collect();
                LatentShapes::from_options(&Options::List(l(entries))).unwrap()
            } else {
                shapes
            };
            let state = WindowingState {
                latents: l([working, 4, 12]
                    .into_iter()
                    .map(|n| data(&[1, 1, n, 2, 2], dev, Kind::Float, 0))
                    .collect()),
                guide_latents: l(vec![None; 3]),
                guide_entries: l(vec![None; 3]),
                keyframe_idxs: l(vec![None; 3]),
                latent_shapes: Some(shapes.clone()),
                dim: 2,
                is_multimodal: p["multimodal"].as_bool().unwrap(),
                temporal_downscale_ratio: 1,
            };
            let window =
                IndexWindow::new(l(vec![0, 2, 3]), 2, 8, None, p["overlap"].as_i64().unwrap())
                    .unwrap();
            let mapped = l(vec![
                l(vec![0, 2, 3]),
                l(if p["action"] == "empty-child" {
                    vec![]
                } else {
                    vec![0, 2]
                }),
                l(vec![2, 4, 5]),
            ]);
            let mut mapper = Mapper {
                action: p["action"].as_str().unwrap().into(),
                original: shapes.clone(),
                window: window.clone(),
                mapped: mapped.clone(),
                calls: vec![],
            };
            let result = state.prepare_window(&window, &mut mapper).map(|out| {
                assert_eq!(
                    case["owners"]["same_window"],
                    Rc::ptr_eq(&out, &window),
                    "{note}"
                );
                assert_eq!(
                    case["owners"]["same_indices"],
                    Rc::ptr_eq(&out.borrow().indices, &window.borrow().indices),
                    "{note}"
                );
                let mut children = vec![];
                let mut aliases = vec![];
                if state.is_multimodal {
                    for m in 1..3 {
                        let child = IndexWindow::modality(&out, m).unwrap();
                        aliases.push(Rc::ptr_eq(
                            &child.borrow().indices,
                            &mapped.borrow()[m as usize],
                        ));
                        children.push(snapshot(&child));
                    }
                }
                assert_eq!(json!(aliases), case["owners"]["child_indices"], "{note}");
                V::Tuple(vec![snapshot(&out), V::List(children)])
            });
            assert_eq!(json!(mapper.calls), case["calls"], "{note}");
            assert_eq!(
                json!(
                    shapes
                        .entries()
                        .unwrap()
                        .iter()
                        .map(|v| v.values().unwrap())
                        .collect::<Vec<_>>()
                ),
                case["shapes"],
                "{note}"
            );
            (result, window)
        } else {
            let dtype = kind(p["dtype"].as_str().unwrap());
            let multi = p["multi"].as_bool().unwrap();
            let guide = p["guide"].as_str().unwrap();
            let mut latents = vec![data(&[1, 1, 8, 2, 2], dev, dtype, 0)];
            if multi {
                latents.push(data(&[1, 1, 4, 2, 2], dev, dtype, 5));
            }
            let entry = |t, h, count| {
                V::Dict(vec![
                    (
                        "latent_shape".into(),
                        list(vec![t, h, if t == 2 { 1 } else { 2 }]),
                    ),
                    ("pre_filter_count".into(), V::Int(count)),
                ])
            };
            let entries = Options::from_value(V::List(if guide == "empty" {
                vec![]
            } else {
                vec![
                    entry(2, if guide == "bad-height" { 0 } else { 1 }, 1),
                    entry(1, 2, 1),
                ]
            }));
            let mut k = vec![0i64; 12];
            k[0] = 2;
            k[2] = 12;
            let kf = V::tensor(Tensor::from_slice(&k).reshape([1, 3, 2, 2]).to_device(dev));
            let frames = data(
                if guide == "bad-rank" {
                    &[1, 1, 3]
                } else {
                    &[1, 1, 3, 2, 2]
                },
                dev,
                dtype,
                7,
            );
            let mut guides = vec![if guide == "missing-frames" {
                None
            } else {
                Some(frames)
            }];
            let mut entrylist = vec![if guide == "none" { None } else { Some(entries) }];
            let mut keyframes = vec![Some(kf)];
            if multi {
                guides.push(None);
                entrylist.push(None);
                keyframes.push(None);
            }
            let state = WindowingState {
                latents: l(latents),
                guide_latents: l(guides),
                guide_entries: l(entrylist),
                keyframe_idxs: l(keyframes),
                latent_shapes: None,
                dim: 2,
                is_multimodal: multi,
                temporal_downscale_ratio: 2,
            };
            let child = IndexWindow::new(l(vec![0, 2]), 2, 4, None, 0).unwrap();
            let window = IndexWindow::new(
                l(ints(&p["indices"])),
                2,
                8,
                multi.then(|| l(vec![(1, child)])),
                1,
            )
            .unwrap();
            {
                let mut w = window.borrow_mut();
                w.causal_anchor = p["anchor"].as_i64();
                w.guide_frames_indices = l(vec![91]);
                w.guide_overlap_info = l(vec![(91, 91)]);
                w.guide_kf_local_positions = l(vec![91]);
                w.guide_downscale_factors = l(vec![91]);
            }
            let old = {
                let w = window.borrow();
                (
                    w.guide_frames_indices.clone(),
                    w.guide_overlap_info.clone(),
                    w.guide_kf_local_positions.clone(),
                    w.guide_downscale_factors.clone(),
                )
            };
            let result = state
                .slice_for_window(&window, &ints(&p["retain"]), None)
                .and_then(|(sliced, counts)| {
                    let outputs = l(vec![l(sliced.clone()), l(sliced.clone())]);
                    state.strip_guide_frames(&outputs, &counts, &window)?;
                    let mut owners = vec![];
                    for row in outputs.borrow().iter() {
                        let mut aliases = vec![];
                        for (j, v) in row.borrow().iter().enumerate() {
                            aliases.push(
                                native::storage_info(&v.as_tensor()?)?.0
                                    == native::storage_info(&sliced[j].as_tensor()?)?.0,
                            );
                        }
                        owners.push(aliases);
                    }
                    assert_eq!(json!(owners), case["owners"]["output_storage"], "{note}");
                    let values = outputs
                        .borrow()
                        .iter()
                        .map(|v| V::List(v.borrow().clone()))
                        .collect();
                    Ok(V::Tuple(vec![
                        V::List(sliced),
                        list(counts),
                        V::List(values),
                    ]))
                });
            let same = {
                let w = window.borrow();
                vec![
                    Rc::ptr_eq(&old.0, &w.guide_frames_indices),
                    Rc::ptr_eq(&old.1, &w.guide_overlap_info),
                    Rc::ptr_eq(&old.2, &w.guide_kf_local_positions),
                    Rc::ptr_eq(&old.3, &w.guide_downscale_factors),
                ]
            };
            assert_eq!(json!(same), case["owners"]["same_guide_lists"], "{note}");
            check(
                &V::List(vec![
                    list(old.0.borrow().clone()),
                    V::List(
                        old.1
                            .borrow()
                            .iter()
                            .map(|(a, b)| V::Tuple(vec![V::Int(*a), V::Int(*b)]))
                            .collect(),
                    ),
                    list(old.2.borrow().clone()),
                    list(old.3.borrow().clone()),
                ]),
                &case["owners"]["old_guide_lists"],
                dev,
                &note,
            );
            (result, window)
        };
        check(&snapshot(&window), &case["post"], dev, &note);
        match out {
            Ok(v) => {
                assert!(!case["error"].is_string(), "{note}: expected {case}");
                check(&v, &case["output"], dev, &note);
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: unexpected {e}");
                reference_error(e, case);
            }
        }
    }
}
