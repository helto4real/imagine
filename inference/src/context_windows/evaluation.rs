//! Whole context-window evaluation. Foreign model/control/callback operations
//! remain required bindings; no model-family or device policy is substituted.
use super::{
    CombineFrame, ConditionAttributes, Handler, LatentShape, LatentShapes, ModalityIndices,
    ModalityMapper, ResizeModel, ResizingBindings, WindowHandle, WindowList, WindowStateModel,
    WindowingState, error, gap, pack_latents, unpack_latents,
};
use crate::{
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
use std::{any::Any, cell::RefCell, rc::Rc};
use tch::Device;

pub trait ContextModel: ResizeModel + WindowStateModel + Any {
    fn class_name(&self) -> &str;
    fn map_context_window_to_modalities(
        &self,
        indices: WindowList<i64>,
        shapes: LatentShapes,
        dim: i64,
    ) -> Result<ModalityIndices, Error>;
}
struct Mapper<'a>(&'a dyn ContextModel);
impl ModalityMapper for Mapper<'_> {
    fn class_name(&self) -> &str {
        self.0.class_name()
    }
    fn map_context_window_to_modalities(
        &mut self,
        indices: WindowList<i64>,
        shapes: LatentShapes,
        dim: i64,
    ) -> Result<ModalityIndices, Error> {
        self.0
            .map_context_window_to_modalities(indices, shapes, dim)
    }
}
pub struct ContextCall<'a> {
    pub handler: &'a Rc<Handler>,
    pub model: &'a Rc<dyn ContextModel>,
    pub input: &'a V,
    pub conditions: &'a Options,
    pub timestep: &'a V,
    pub options: &'a Options,
}
pub enum ContextCallbackFrame<'a> {
    ExecuteStart {
        call: &'a ContextCall<'a>,
    },
    Evaluate {
        call: &'a ContextCall<'a>,
        window_index: i64,
        window: &'a WindowHandle,
        device: Option<Device>,
        first_device: Option<Device>,
    },
    Combine {
        call: &'a ContextCall<'a>,
        frame: &'a CombineFrame<'a>,
        conditions: &'a Options,
        window_index: i64,
        total_windows: usize,
    },
    ExecuteCleanup {
        call: &'a ContextCall<'a>,
    },
}
pub struct ContextBatch<'a> {
    pub model: &'a Rc<dyn ContextModel>,
    pub conditions: &'a Options,
    pub input: &'a V,
    pub timestep: &'a V,
    pub options: &'a Options,
}
pub trait ContextBindings: ResizingBindings + ConditionAttributes {
    fn throw_if_interrupted(&self) -> Result<(), Error>;
    fn calc_cond_batch(&self, call: &ContextBatch<'_>) -> Result<WindowList<V>, Error>;
    /// The enum represents the original fixed callback arguments and their
    /// shared mutable containers. Evaluate's options is passed twice in Python;
    /// both positions refer to this same `call.options` owner.
    fn callback(&self, callback: &Options, frame: ContextCallbackFrame<'_>) -> Result<(), Error>;
    fn evaluation_locals(&self, _: &EvaluationLocals, _: &str) -> Result<(), Error> {
        Ok(())
    }
}
pub struct ContextResult {
    pub window_index: i64,
    pub outputs: WindowList<WindowList<V>>,
    pub conditions: Options,
    pub window: WindowHandle,
}
pub struct EvaluationLocals {
    pub primary: V,
    pub window: Option<WindowHandle>,
    pub callback: Option<Rc<Options>>,
    pub sliced: Option<Vec<V>>,
    pub guide_counts: Option<Vec<i64>>,
    pub sub_input: Option<V>,
    pub sub_shapes: Option<LatentShapes>,
    pub sub_timestep: Option<V>,
    pub sub_conditions: Option<Options>,
    pub model_outputs: Option<WindowList<V>>,
    pub outputs: Option<WindowList<WindowList<V>>>,
    pub last_primary_output: Option<V>,
    pub results: WindowList<Rc<ContextResult>>,
}
pub struct EvaluationRequest<'a> {
    pub call: ContextCall<'a>,
    pub windows: &'a WindowList<(i64, WindowHandle)>,
    pub state: &'a WindowingState,
    pub total_windows: Option<i64>,
    pub device: Option<Device>,
    pub first_device: Option<Device>,
}
fn at<T: Clone>(list: &WindowList<T>, index: usize) -> Result<T, Error> {
    list.borrow()
        .get(index)
        .cloned()
        .ok_or_else(|| error("IndexError", "list index out of range"))
}
pub(super) fn required(value: &Options, key: &str) -> Result<Options, Error> {
    match value {
        Options::Dict(_) => value
            .get(key)?
            .ok_or_else(|| error("KeyError", format!("'{key}'"))),
        Options::List(_) | Options::Tuple(_) => Err(error(
            "TypeError",
            format!(
                "{} indices must be integers or slices, not str",
                value.typename()
            ),
        )),
        Options::Atom(V::Text(_)) => Err(error(
            "TypeError",
            "string indices must be integers, not 'str'",
        )),
        Options::Atom(V::None | V::Bool(_) | V::Int(_) | V::UInt(_) | V::Float(_)) => Err(error(
            "TypeError",
            format!("'{}' object is not subscriptable", value.typename()),
        )),
        _ => Err(gap(
            "context options subscription requires its original value binding",
        )),
    }
}
impl Handler {
    pub fn evaluate_context_windows(
        bindings: &dyn ContextBindings,
        request: EvaluationRequest<'_>,
    ) -> Result<WindowList<Rc<ContextResult>>, Error> {
        let call = &request.call;
        let handler = call.handler;
        let state = request.state;
        let mut locals = EvaluationLocals {
            primary: at(&state.latents, 0)?,
            window: None,
            callback: None,
            sliced: None,
            guide_counts: None,
            sub_input: None,
            sub_shapes: None,
            sub_timestep: None,
            sub_conditions: None,
            model_outputs: None,
            outputs: None,
            last_primary_output: None,
            results: Rc::new(RefCell::new(vec![])),
        };
        let mut position = 0;
        loop {
            let Some((index, window)) = request.windows.borrow().get(position).cloned() else {
                break;
            };
            position += 1;
            locals.window = Some(window.clone());
            bindings.evaluation_locals(&locals, "interrupt")?;
            bindings.throw_if_interrupted()?;
            bindings.evaluation_locals(&locals, "prepare_window")?;
            let window = state.prepare_window(&window, &mut Mapper(call.model.as_ref()))?;
            locals.window = Some(window.clone());
            let mut anchor_applied = false;
            if handler.settings.borrow().causal_window_fix {
                let first = window
                    .borrow()
                    .indices
                    .borrow()
                    .first()
                    .copied()
                    .ok_or_else(|| error("IndexError", "list index out of range"))?;
                let anchor = i128::from(first) - 1;
                if anchor >= 0
                    && anchor
                        < i128::from(native::context_size(
                            &call.input.as_tensor()?,
                            handler.settings.borrow().dim,
                        )?)
                {
                    window.borrow_mut().causal_anchor = Some(anchor as i64);
                    anchor_applied = true;
                }
            }
            let (sliced, counts) = {
                let retain = handler.settings.borrow().latent_retain.clone();
                state.slice_for_window(&window, &retain.borrow(), request.device)?
            };
            locals.sliced = Some(sliced);
            locals.guide_counts = Some(counts);
            let callbacks = handler.callbacks.borrow().all("evaluate_context_windows");
            for callback in &callbacks {
                let old = locals.callback.replace(callback.clone());
                drop(old);
                bindings.evaluation_locals(&locals, "callback")?;
                bindings.callback(
                    callback,
                    ContextCallbackFrame::Evaluate {
                        call,
                        window_index: index,
                        window: &window,
                        device: request.device,
                        first_device: request.first_device,
                    },
                )?;
            }
            drop(callbacks);
            let (first, last) = {
                let w = window.borrow();
                let values = w.indices.borrow();
                (
                    *values
                        .first()
                        .ok_or_else(|| error("IndexError", "list index out of range"))?,
                    *values
                        .last()
                        .ok_or_else(|| error("IndexError", "list index out of range"))?,
                )
            };
            let frames = super::state::shape_at(
                &locals.primary.as_tensor()?.size(),
                handler.settings.borrow().dim,
            )?;
            let total = request
                .total_windows
                .filter(|n| *n != 0)
                .map(i128::from)
                .unwrap_or_else(|| request.windows.borrow().len() as i128);
            let guides = locals.guide_counts.as_ref().unwrap()[0];
            let message = format!(
                "Context window {}/{total}: frames {first}-{last} of {frames}{}",
                i128::from(index) + 1,
                if guides > 0 {
                    format!(" (+{guides} guide frames)")
                } else {
                    String::new()
                }
            );
            bindings.evaluation_locals(&locals, "info")?;
            bindings.info(&message)?;
            let sliced = locals.sliced.as_ref().unwrap();
            let (sub_input, sub_shapes) = if state.is_multimodal && sliced.len() > 1 {
                pack_latents(sliced)?
            } else {
                let value = sliced
                    .first()
                    .cloned()
                    .ok_or_else(|| error("IndexError", "list index out of range"))?;
                let shapes =
                    LatentShapes::new(vec![LatentShape::Size(Rc::new(value.as_tensor()?.size()))]);
                (value, shapes)
            };
            locals.sub_input = Some(sub_input);
            locals.sub_shapes = Some(sub_shapes);
            required(call.options, "transformer_options")?
                .set("context_window", Options::Object(window.clone()))?;
            locals.sub_timestep = Some(window.borrow().get_tensor(
                call.timestep,
                None,
                Some(0),
                &[],
            )?);
            let mut sub_conditions = vec![];
            let mut ci = 0;
            while let Some(condition) = super::metadata::next(call.conditions, &mut ci)? {
                sub_conditions.push(handler.get_resized_cond(
                    bindings,
                    &condition,
                    &locals.primary,
                    &window,
                    None,
                )?);
            }
            locals.sub_conditions = Some(Options::List(Rc::new(RefCell::new(sub_conditions))));
            state.patch_latent_shapes(
                locals.sub_conditions.as_ref().unwrap(),
                &locals.sub_shapes.as_ref().unwrap().as_options(),
            )?;
            bindings.evaluation_locals(&locals, "model")?;
            locals.model_outputs = Some(bindings.calc_cond_batch(&ContextBatch {
                model: call.model,
                conditions: locals.sub_conditions.as_ref().unwrap(),
                input: locals.sub_input.as_ref().unwrap(),
                timestep: locals.sub_timestep.as_ref().unwrap(),
                options: call.options,
            })?);
            let raw_outputs = locals.model_outputs.as_ref().unwrap();
            let length = raw_outputs.borrow().len();
            let mut outputs = vec![];
            for ci in 0..length {
                outputs.push(Rc::new(RefCell::new(unpack_latents(
                    &at(raw_outputs, ci)?,
                    locals.sub_shapes.as_ref().unwrap(),
                )?)));
            }
            locals.outputs = Some(Rc::new(RefCell::new(outputs)));
            let outputs = locals.outputs.as_ref().unwrap();
            if anchor_applied {
                for ci in 0..outputs.borrow().len() {
                    let row = at(outputs, ci)?;
                    let value = at(&row, 0)?;
                    locals.last_primary_output = Some(value.clone());
                    let tensor = value.as_tensor()?;
                    let dim = handler.settings.borrow().dim;
                    row.borrow_mut()[0] = V::tensor(tensor.f_narrow(
                        dim,
                        1,
                        super::state::shape_at(&tensor.size(), dim)? - 1,
                    )?);
                }
            }
            state.strip_guide_frames(outputs, locals.guide_counts.as_ref().unwrap(), &window)?;
            locals.results.borrow_mut().push(Rc::new(ContextResult {
                window_index: index,
                outputs: outputs.clone(),
                conditions: locals.sub_conditions.as_ref().unwrap().clone(),
                window,
            }));
        }
        bindings.evaluation_locals(&locals, "return")?;
        Ok(locals.results.clone())
    }
}
