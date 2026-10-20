//! The source handler's complete window execution and finalization. The finally
//! block deliberately starts after evaluation/combination, as in the reference.
use super::{
    CombineFrame, ContextBatch, ContextBindings, ContextCall, ContextCallbackFrame, ContextResult,
    EvaluationLocals, EvaluationRequest, Fuse, Handler, IndexWindow, ResizeLocals, ResizeModelSlot,
    ResizingBindings, WindowHandle, WindowList, WindowingState, error, pack_latents, shape_for_dim,
};
use crate::{
    execution::Runtime, execution_value::Value as V, native::Error, patching::options::Options,
};
use std::{cell::RefCell, rc::Rc};
use tch::Tensor;

pub trait ExecutionBindings: ContextBindings {
    fn warning(&self, message: &str) -> Result<(), Error>;
    fn execution_locals(&self, _: &ExecutionLocals, _: &str) -> Result<(), Error> {
        Ok(())
    }
}
pub struct ExecutionLocals {
    pub state: WindowingState,
    pub windows: WindowList<WindowHandle>,
    pub enumerated: WindowList<(i64, WindowHandle)>,
    pub accum: WindowList<WindowList<V>>,
    pub counts: WindowList<WindowList<V>>,
    pub biases: WindowList<WindowList<WindowList<f64>>>,
    pub callback: Option<Rc<Options>>,
    pub results: Option<WindowList<Rc<ContextResult>>>,
    pub result: Option<Rc<ContextResult>>,
    pub modality_outputs: Option<WindowList<V>>,
    pub modality_window: Option<WindowHandle>,
    pub result_out: Option<WindowList<V>>,
    pub finalized: Option<WindowList<V>>,
    pub last_final: Option<V>,
    pub packed: Option<V>,
}
fn list<T>(items: Vec<T>) -> WindowList<T> {
    Rc::new(RefCell::new(items))
}
fn at<T: Clone>(items: &WindowList<T>, i: usize) -> Result<T, Error> {
    items
        .borrow()
        .get(i)
        .cloned()
        .ok_or_else(|| error("IndexError", "list index out of range"))
}
fn len(value: &Options) -> Result<usize, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().len()),
        Options::Tuple(v) => Ok(v.len()),
        _ => Err(error(
            "TypeError",
            format!("object of type '{}' has no len()", value.typename()),
        )),
    }
}
/// Keep access to the outer source frame through nested evaluation calls. This
/// forwards every real binding; observers do not stand in for those operations.
struct DuringEvaluation<'a> {
    bindings: &'a dyn ExecutionBindings,
    locals: &'a ExecutionLocals,
}
impl super::ConditionAttributes for DuringEvaluation<'_> {
    fn cond(&self, value: &Options) -> Result<Options, Error> {
        self.bindings.cond(value)
    }
}
impl ResizingBindings for DuringEvaluation<'_> {
    fn prepare_control_objects(
        &self,
        control: &Options,
        device: Option<tch::Device>,
    ) -> Result<Options, Error> {
        self.bindings.prepare_control_objects(control, device)
    }
    fn info(&self, message: &str) -> Result<(), Error> {
        self.bindings.info(message)
    }
    fn source_locals(&self, locals: &ResizeLocals, site: &str) -> Result<(), Error> {
        self.bindings.source_locals(locals, site)
    }
}
impl ContextBindings for DuringEvaluation<'_> {
    fn throw_if_interrupted(&self) -> Result<(), Error> {
        self.bindings.throw_if_interrupted()
    }
    fn calc_cond_batch(&self, call: &ContextBatch<'_>) -> Result<WindowList<V>, Error> {
        self.bindings.execution_locals(self.locals, "model")?;
        self.bindings.calc_cond_batch(call)
    }
    fn callback(&self, callback: &Options, frame: ContextCallbackFrame<'_>) -> Result<(), Error> {
        self.bindings
            .execution_locals(self.locals, "evaluate-callback")?;
        self.bindings.callback(callback, frame)
    }
    fn evaluation_locals(&self, locals: &EvaluationLocals, site: &str) -> Result<(), Error> {
        self.bindings.evaluation_locals(locals, site)
    }
}
impl Handler {
    pub fn execute(
        runtime: &Runtime,
        bindings: &dyn ExecutionBindings,
        call: ContextCall<'_>,
    ) -> Result<WindowList<V>, Error> {
        let handler = call.handler;
        let old = std::mem::replace(
            &mut *handler.model.borrow_mut(),
            ResizeModelSlot::Bound(call.model.clone()),
        );
        drop(old);
        let transformer = super::evaluation::required(call.options, "transformer_options")?;
        handler.set_step(
            call.timestep,
            &super::evaluation::required(&transformer, "sample_sigmas")?.snapshot()?,
        )?;
        drop(transformer);
        let state = handler.build_window_state(
            call.input,
            call.conditions,
            bindings,
            call.model.as_ref(),
        )?;
        let modalities = state.latents.borrow().len();
        let windows = list(
            handler.get_context_windows(&at(&state.latents, 0)?, &mut |message| {
                bindings.warning(message)
            })?,
        );
        let enumerated = list(
            windows
                .borrow()
                .iter()
                .enumerate()
                .map(|(i, window)| (i as i64, window.clone()))
                .collect(),
        );
        let total = enumerated.borrow().len();
        // Each comprehension has its own iterator and releases its loop values
        // when complete. Python's subsequent named locals live in `locals`.
        let mut accum = vec![];
        let mut position = 0;
        while let Some(m) = state.latents.borrow().get(position).cloned() {
            position += 1;
            let mut row = vec![];
            let mut ci = 0;
            while super::metadata::next(call.conditions, &mut ci)?.is_some() {
                row.push(V::tensor(m.as_tensor()?.f_zeros_like()?));
            }
            accum.push(list(row));
        }
        let mut counts = vec![];
        let mut position = 0;
        let relative = matches!(handler.settings.borrow().fuse, Fuse::Relative);
        while let Some(m) = state.latents.borrow().get(position).cloned() {
            position += 1;
            let mut row = vec![];
            let mut ci = 0;
            while super::metadata::next(call.conditions, &mut ci)?.is_some() {
                let tensor = m.as_tensor()?;
                let shape = shape_for_dim(&tensor, handler.settings.borrow().dim)?;
                let options = (runtime.native.default_dtype()?, tensor.device());
                row.push(V::tensor(if relative {
                    Tensor::f_ones(&shape, options)?
                } else {
                    Tensor::f_zeros(&shape, options)?
                }));
            }
            counts.push(list(row));
        }
        let mut biases = vec![];
        let mut position = 0;
        while let Some(m) = state.latents.borrow().get(position).cloned() {
            position += 1;
            let mut row = vec![];
            let mut ci = 0;
            while super::metadata::next(call.conditions, &mut ci)?.is_some() {
                let length =
                    super::state::shape_at(&m.as_tensor()?.size(), handler.settings.borrow().dim)?;
                row.push(list(vec![0.; length as usize]));
            }
            biases.push(list(row));
        }
        let mut locals = ExecutionLocals {
            state,
            windows,
            enumerated,
            accum: list(accum),
            counts: list(counts),
            biases: list(biases),
            callback: None,
            results: None,
            result: None,
            modality_outputs: None,
            modality_window: None,
            result_out: None,
            finalized: None,
            last_final: None,
            packed: None,
        };
        let callbacks = handler.callbacks.borrow().all("execute_start");
        for callback in &callbacks {
            let old = locals.callback.replace(callback.clone());
            drop(old);
            bindings.execution_locals(&locals, "start-callback")?;
            bindings.callback(callback, ContextCallbackFrame::ExecuteStart { call: &call })?;
        }
        drop(callbacks);
        let mut position = 0;
        loop {
            let Some(enum_window) = locals.enumerated.borrow().get(position).cloned() else {
                break;
            };
            position += 1;
            let results = Self::evaluate_context_windows(
                &DuringEvaluation {
                    bindings,
                    locals: &locals,
                },
                EvaluationRequest {
                    call: ContextCall { ..call },
                    windows: &list(vec![enum_window]),
                    state: &locals.state,
                    total_windows: Some(total as i64),
                    device: None,
                    first_device: None,
                },
            )?;
            locals.results = Some(results.clone());
            let mut ri = 0;
            loop {
                let Some(result) = results.borrow().get(ri).cloned() else {
                    break;
                };
                ri += 1;
                locals.result = Some(result.clone());
                for modality in 0..modalities {
                    let mut outputs = vec![];
                    for ci in 0..len(call.conditions)? {
                        outputs.push(at(&at(&result.outputs, ci)?, modality)?);
                    }
                    locals.modality_outputs = Some(list(outputs));
                    locals.modality_window =
                        Some(IndexWindow::modality(&result.window, modality as i64)?);
                    let input = at(&locals.state.latents, modality)?;
                    let accum = at(&locals.accum, modality)?;
                    let counts = at(&locals.counts, modality)?;
                    let biases = at(&locals.biases, modality)?;
                    handler.combine(
                        runtime,
                        &CombineFrame {
                            input: &input,
                            outputs: locals.modality_outputs.as_ref().unwrap(),
                            window: locals.modality_window.as_ref().unwrap(),
                            accum: &accum,
                            counts: &counts,
                            biases: &biases,
                        },
                        &mut |_, frame, _| {
                            let callbacks = handler
                                .callbacks
                                .borrow()
                                .all("combine_context_window_results");
                            // A separate source frame owns this last callback.
                            let mut last_callback = None;
                            for callback in &callbacks {
                                let old = last_callback.replace(callback.clone());
                                drop(old);
                                bindings.execution_locals(&locals, "combine-callback")?;
                                bindings.callback(
                                    callback,
                                    ContextCallbackFrame::Combine {
                                        call: &call,
                                        frame,
                                        conditions: &result.conditions,
                                        window_index: result.window_index,
                                        total_windows: total,
                                    },
                                )?;
                            }
                            drop(callbacks);
                            drop(last_callback);
                            Ok(())
                        },
                    )?;
                }
            }
        }
        // Only this finalization block is protected by the reference finally.
        let outcome = (|| {
            locals.result_out = Some(list(vec![]));
            for ci in 0..len(call.conditions)? {
                locals.finalized = Some(list(vec![]));
                for modality in 0..modalities {
                    if !matches!(handler.settings.borrow().fuse, Fuse::Relative) {
                        let mut target = at(&at(&locals.accum, modality)?, ci)?.as_tensor()?;
                        let count = at(&at(&locals.counts, modality)?, ci)?.as_tensor()?;
                        let _ = target.f_div_(&count)?;
                    }
                    locals.last_final = Some(at(&at(&locals.accum, modality)?, ci)?);
                    if let Some(guide) = at(&locals.state.guide_latents, modality)? {
                        let joined = Tensor::f_cat(
                            &[
                                locals.last_final.as_ref().unwrap().as_tensor()?,
                                guide.as_tensor()?,
                            ],
                            handler.settings.borrow().dim,
                        )?;
                        locals.last_final = Some(V::tensor(joined));
                    }
                    locals
                        .finalized
                        .as_ref()
                        .unwrap()
                        .borrow_mut()
                        .push(locals.last_final.as_ref().unwrap().clone());
                }
                let finalized = locals.finalized.as_ref().unwrap();
                locals.packed = Some(
                    if locals.state.is_multimodal && finalized.borrow().len() > 1 {
                        pack_latents(&finalized.borrow())?.0
                    } else {
                        at(finalized, 0)?
                    },
                );
                locals
                    .result_out
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .push(locals.packed.as_ref().unwrap().clone());
            }
            Ok(locals.result_out.as_ref().unwrap().clone())
        })();
        let callbacks = handler.callbacks.borrow().all("execute_cleanup");
        for callback in &callbacks {
            let old = locals.callback.replace(callback.clone());
            drop(old);
            bindings.execution_locals(&locals, "cleanup-callback")?;
            bindings.callback(
                callback,
                ContextCallbackFrame::ExecuteCleanup { call: &call },
            )?;
        }
        drop(callbacks);
        if outcome.is_ok() {
            bindings.execution_locals(&locals, "return")?;
        }
        outcome
    }
}
