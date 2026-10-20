//! Shared handler step tracking, window construction and accumulation bodies.
//! Whole execution, model/conditioning and callback-registry binding are separate.
use super::resizing::ResizeModelSlot;
use super::{
    Fuse, IndexWindow, Schedule, ScheduleSettings, WindowHandle, WindowList, error, gap,
    match_weights_to_dim,
};
use crate::patching::{extensions::OrderedRegistry, options::Options};
use crate::sampling::locals::SourceLocals;
use crate::{
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Clone)]
pub struct HandlerSettings {
    pub schedule: Schedule,
    pub fuse: Fuse,
    pub length: i64,
    pub overlap: i64,
    pub stride: i64,
    pub closed_loop: bool,
    pub dim: i64,
    pub freenoise: bool,
    pub split_conditions: bool,
    pub causal_window_fix: bool,
    pub condition_retain: WindowList<i64>,
    pub latent_retain: WindowList<i64>,
}
pub struct Handler {
    pub settings: RefCell<HandlerSettings>,
    pub step: Cell<u64>,
    pub callbacks: Rc<RefCell<OrderedRegistry<Options>>>,
    pub model: RefCell<ResizeModelSlot>,
}
pub struct CombineFrame<'a> {
    pub input: &'a V,
    pub outputs: &'a WindowList<V>,
    pub window: &'a WindowHandle,
    pub accum: &'a WindowList<V>,
    pub counts: &'a WindowList<V>,
    pub biases: &'a WindowList<WindowList<f64>>,
}
fn item<T: Clone>(list: &WindowList<T>, i: usize) -> Result<T, Error> {
    list.borrow()
        .get(i)
        .cloned()
        .ok_or_else(|| error("IndexError", "list index out of range"))
}
fn position(index: i64, length: usize) -> Result<usize, Error> {
    let v = if index < 0 {
        length as i128 + i128::from(index)
    } else {
        i128::from(index)
    };
    usize::try_from(v)
        .ok()
        .filter(|v| *v < length)
        .ok_or_else(|| error("IndexError", "list index out of range"))
}
impl Handler {
    pub fn new(schedule: Schedule, fuse: Fuse) -> Self {
        Self {
            settings: RefCell::new(HandlerSettings {
                schedule,
                fuse,
                length: 1,
                overlap: 0,
                stride: 1,
                closed_loop: false,
                dim: 0,
                freenoise: false,
                split_conditions: false,
                causal_window_fix: true,
                condition_retain: Rc::new(RefCell::new(vec![])),
                latent_retain: Rc::new(RefCell::new(vec![])),
            }),
            step: Cell::new(0),
            callbacks: Rc::new(RefCell::new(OrderedRegistry::default())),
            model: RefCell::new(ResizeModelSlot::Missing),
        }
    }
    /// Original Tensor operations after extracting transformer_options.sample_sigmas.
    pub fn set_step(&self, timestep: &V, sample_sigmas: &V) -> Result<(), Error> {
        let sigmas = sample_sigmas.as_tensor()?;
        let current =
            native::context_scalar_index(&timestep.as_tensor()?, &[0])?.f_to_kind(sigmas.kind())?;
        let matches = sigmas
            .f_isclose(&current, 0.0001, 1e-8, false)?
            .f_nonzero()?;
        if matches.numel() == 0 {
            return Ok(());
        }
        let index = native::context_scalar_index(&matches, &[0])?.f_int64_value(&[])?;
        self.step.set(index as u64);
        Ok(())
    }
    pub fn get_context_windows(
        &self,
        input: &V,
        warning: &mut dyn FnMut(&str) -> Result<(), Error>,
    ) -> Result<Vec<WindowHandle>, Error> {
        let settings = self.settings.borrow().clone();
        let length = native::context_size(&input.as_tensor()?, settings.dim)?;
        let indices = settings.schedule.windows(
            length,
            ScheduleSettings {
                length: settings.length,
                overlap: settings.overlap,
                stride: settings.stride,
                closed_loop: settings.closed_loop,
                step: self.step.get(),
            },
            warning,
        )?;
        indices
            .into_iter()
            .map(|indices| {
                IndexWindow::new(
                    Rc::new(RefCell::new(indices)),
                    settings.dim,
                    length,
                    None,
                    settings.overlap,
                )
            })
            .collect()
    }
    pub fn combine(
        &self,
        runtime: &Runtime,
        frame: &CombineFrame<'_>,
        after: &mut dyn FnMut(&Self, &CombineFrame<'_>, &SourceLocals) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let settings = self.settings.borrow().clone();
        let full = frame.input.as_tensor()?;
        let mut locals = SourceLocals::default();
        if matches!(settings.fuse, Fuse::Relative) {
            let indices = frame.window.borrow().indices.borrow().clone();
            for (pos, index) in indices.iter().enumerate() {
                let first = *indices
                    .first()
                    .ok_or_else(|| error("IndexError", "list index out of range"))?;
                let last = *indices.last().unwrap();
                let center =
                    crate::python_numbers::integer_ratio(i128::from(first) + i128::from(last), 2)?;
                let denominator = ((i128::from(last) - i128::from(first)) as f64 + 1e-2) / 2.;
                let bias = 1. - (*index as f64 - center).abs() / denominator;
                let bias = if bias > 1e-2 { bias } else { 1e-2 };
                for i in 0..frame.outputs.borrow().len() {
                    let biases = item(frame.biases, i)?;
                    let index_in_bias = position(*index, biases.borrow().len())?;
                    let total = biases.borrow()[index_in_bias];
                    let sum = total + bias;
                    if sum == 0. {
                        return Err(error("ZeroDivisionError", "float division by zero"));
                    }
                    let old_weight = total / sum;
                    let new_weight = bias / sum;
                    let accum = item(frame.accum, i)?.as_tensor()?;
                    let previous = native::context_scalar_at(&accum, *index, settings.dim)?
                        .f_mul_scalar(old_weight)?;
                    let next = native::context_scalar_at(
                        &item(frame.outputs, i)?.as_tensor()?,
                        pos as i64,
                        settings.dim,
                    )?
                    .f_mul_scalar(new_weight)?;
                    let updated = previous.f_add(&next)?;
                    drop(previous);
                    drop(next);
                    native::context_scalar_at_put(&accum, *index, settings.dim, &updated)?;
                    drop(updated);
                    biases.borrow_mut()[index_in_bias] = sum;
                }
            }
        } else {
            let shape = full.size();
            let dim = if settings.dim < 0 {
                shape.len() as i128 + i128::from(settings.dim)
            } else {
                i128::from(settings.dim)
            };
            let full_length = usize::try_from(dim)
                .ok()
                .and_then(|v| shape.get(v))
                .copied()
                .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
            let w = frame.window.borrow();
            let weights = settings.fuse.weights(
                &runtime.native,
                i64::try_from(w.context_length)
                    .map_err(|_| gap("window length exceeds native i64"))?,
                full_length,
                &w.indices.borrow(),
                w.overlap,
            )?;
            drop(w);
            if matches!(weights, V::Tensor(_)) {
                let _ = locals.keep("weights", weights.as_tensor()?);
            }
            let weights = locals.keep(
                "weights_tensor",
                match_weights_to_dim(runtime, &weights, &full, settings.dim, Some(full.device()))?,
            );
            for i in 0..frame.outputs.borrow().len() {
                let value = V::tensor(item(frame.outputs, i)?.as_tensor()?.f_mul(&weights)?);
                frame
                    .window
                    .borrow()
                    .add_window(&item(frame.accum, i)?, &value, None)?;
                drop(value);
                frame.window.borrow().add_window(
                    &item(frame.counts, i)?,
                    &V::tensor(weights.shallow_clone()),
                    None,
                )?;
            }
        }
        after(self, frame, &locals)
    }
}
