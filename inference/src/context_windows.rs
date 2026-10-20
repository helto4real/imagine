//! Shared context schedules, fusion weights and FreeNoise from the locked core.
//! Handler/controller and multimodal conditioning bindings remain separate.
use crate::{
    execution::Runtime,
    execution_value::Value as V,
    native::{Error, Kitchen},
};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Tensor};
mod build_state;
mod conditioning;
mod evaluation;
mod execute;
mod freenoise;
mod handler;
mod latents;
mod metadata;
mod resizing;
mod shapes;
mod state;
mod wrappers;
pub use build_state::{WindowBuildLocals, WindowStateModel};
pub use conditioning::{GuideOverlap, compute_guide_overlap, slice_condition};
pub use evaluation::{
    ContextBatch, ContextBindings, ContextCall, ContextCallbackFrame, ContextModel, ContextResult,
    EvaluationLocals, EvaluationRequest,
};
pub use execute::{ExecutionBindings, ExecutionLocals};
pub use handler::{CombineFrame, Handler, HandlerSettings};
pub use latents::{pack_latents, unpack_latents};
pub use metadata::{BuiltinConditionAttributes, ConditionAttributes};
pub use resizing::{ResizeLocals, ResizeModel, ResizeModelSlot, ResizingBindings};
pub use shapes::{LatentShape, LatentShapes};
pub use state::{ModalityIndices, ModalityMapper, WindowingState};
pub use wrappers::{
    ContextWrapperModel, ContextWrappers, SamplerWrapperBindings, prepare_sampling_shape,
    prepare_sampling_wrapper, sampler_sample_wrapper,
};
pub type WindowHandle = Rc<RefCell<IndexWindow>>;
pub type ModalityWindows = Rc<RefCell<Vec<(i64, WindowHandle)>>>;
pub type WindowList<T> = Rc<RefCell<Vec<T>>>;
pub struct IndexWindow {
    pub indices: Rc<RefCell<Vec<i64>>>,
    pub context_length: usize,
    pub overlap: i64,
    pub dim: i64,
    pub total_frames: i64,
    pub center_ratio: f64,
    pub modality_windows: Option<ModalityWindows>,
    pub causal_anchor: Option<i64>,
    pub guide_frames_indices: WindowList<i64>,
    pub guide_overlap_info: WindowList<(i64, i64)>,
    pub guide_kf_local_positions: WindowList<i64>,
    pub guide_downscale_factors: WindowList<i64>,
}
impl IndexWindow {
    pub fn new(
        indices: Rc<RefCell<Vec<i64>>>,
        dim: i64,
        total_frames: i64,
        modality_windows: Option<ModalityWindows>,
        overlap: i64,
    ) -> Result<WindowHandle, Error> {
        let count = indices.borrow().len();
        let min = *indices
            .borrow()
            .iter()
            .min()
            .ok_or_else(|| error("ValueError", "min() iterable argument is empty"))?;
        let max = *indices.borrow().iter().max().unwrap();
        let center_ratio = crate::python_numbers::integer_ratio(
            i128::from(min) + i128::from(max),
            2 * i128::from(total_frames),
        )?;
        Ok(Rc::new(RefCell::new(Self {
            indices,
            context_length: count,
            overlap,
            dim,
            total_frames,
            center_ratio,
            modality_windows,
            causal_anchor: None,
            guide_frames_indices: Rc::new(RefCell::new(vec![])),
            guide_overlap_info: Rc::new(RefCell::new(vec![])),
            guide_kf_local_positions: Rc::new(RefCell::new(vec![])),
            guide_downscale_factors: Rc::new(RefCell::new(vec![])),
        })))
    }
    pub fn get_tensor(
        &self,
        full: &V,
        device: Option<Device>,
        dim: Option<i64>,
        retain: &[i64],
    ) -> Result<V, Error> {
        let dim = dim.unwrap_or(self.dim);
        let tensor = full.as_tensor()?;
        if dim == 0 {
            let size = *tensor
                .size()
                .first()
                .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
            if size == 1 {
                return Ok(full.clone());
            }
        }
        let mut indices = self.indices.borrow().clone();
        if let Some(anchor) = self.causal_anchor
            && anchor >= 0
        {
            indices.insert(0, anchor);
        }
        let window = crate::native::context_index(&tensor, &indices, dim)?;
        if !retain.is_empty() {
            let source = crate::native::context_index(&tensor, retain, dim)?;
            crate::native::context_index_put(&window, retain, dim, &source)?;
        }
        Ok(V::tensor(if let Some(device) = device {
            window.f_to_device(device)?
        } else {
            window
        }))
    }
    pub fn add_window(&self, full: &V, to_add: &V, dim: Option<i64>) -> Result<V, Error> {
        let dim = dim.unwrap_or(self.dim);
        let tensor = full.as_tensor()?;
        let indices = self.indices.borrow().clone();
        let mut window = crate::native::context_index(&tensor, &indices, dim)?;
        let _ = window.f_add_(&to_add.as_tensor()?)?;
        crate::native::context_index_put(&tensor, &indices, dim, &window)?;
        Ok(full.clone())
    }
    pub fn region_index(&self, regions: i64) -> Result<i64, Error> {
        let value = self.center_ratio * regions as f64;
        if !value.is_finite() {
            return Err(error(
                if value.is_nan() {
                    "ValueError"
                } else {
                    "OverflowError"
                },
                if value.is_nan() {
                    "cannot convert float NaN to integer"
                } else {
                    "cannot convert float infinity to integer"
                },
            ));
        }
        let maximum = regions
            .checked_sub(1)
            .ok_or_else(|| gap("region result exceeds native i64"))?;
        Ok((value.trunc() as i128).max(0).min(i128::from(maximum)) as i64)
    }
    pub fn modality(this: &WindowHandle, index: i64) -> Result<WindowHandle, Error> {
        if index == 0 {
            return Ok(this.clone());
        }
        this.borrow()
            .modality_windows
            .as_ref()
            .ok_or_else(|| error("TypeError", "'NoneType' object is not subscriptable"))?
            .borrow()
            .iter()
            .find(|(key, _)| *key == index)
            .map(|(_, window)| window.clone())
            .ok_or_else(|| error("KeyError", index.to_string()))
    }
}
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn gap(message: &str) -> Error {
    Error::BindingGap {
        backend: "context_windows".into(),
        operation: message.into(),
    }
}
fn range(start: i128, stop: i128, step: i128) -> Result<Vec<i64>, Error> {
    if step == 0 {
        return Err(error("ValueError", "range() arg 3 must not be zero"));
    }
    let count = if step > 0 {
        ((stop - start - 1).div_euclid(step) + 1).max(0)
    } else {
        ((start - stop - 1).div_euclid(-step) + 1).max(0)
    };
    let count = usize::try_from(count).map_err(|_| error("MemoryError", ""))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| error("MemoryError", ""))?;
    for i in 0..count {
        result.push(
            i64::try_from(start + step * i as i128)
                .map_err(|_| gap("context range value exceeds native i64 admission"))?,
        );
    }
    Ok(result)
}
fn modulo(value: i128, divisor: i64) -> Result<i64, Error> {
    if divisor == 0 {
        return Err(error("ZeroDivisionError", "integer modulo by zero"));
    }
    let divisor = i128::from(divisor);
    let rem = value % divisor;
    Ok((if rem != 0 && (rem < 0) != (divisor < 0) {
        rem + divisor
    } else {
        rem
    }) as i64)
}
pub fn missing_indexes(windows: &[Vec<i64>], frames: i64) -> Result<Vec<i64>, Error> {
    let mut out = range(0, frames.into(), 1)?;
    for window in windows {
        for value in window {
            if let Some(index) = out.iter().position(|v| v == value) {
                out.remove(index);
            }
        }
    }
    Ok(out)
}
pub fn does_roll_over(window: &[i64], frames: i64) -> Result<(bool, i64), Error> {
    let mut previous = -1;
    for (index, value) in window.iter().enumerate() {
        let value = modulo((*value).into(), frames)?;
        if value < previous {
            return Ok((true, index as i64));
        }
        previous = value;
    }
    Ok((false, -1))
}
pub fn shift_to_start(window: &mut [i64], frames: i64) -> Result<(), Error> {
    let start = *window
        .first()
        .ok_or_else(|| error("IndexError", "list index out of range"))?;
    for value in window {
        *value = modulo(
            i128::from(*value) - i128::from(start) + i128::from(frames),
            frames,
        )?;
    }
    Ok(())
}
pub fn shift_to_end(window: &mut [i64], frames: i64) -> Result<(), Error> {
    shift_to_start(window, frames)?;
    let end = i128::from(*window.last().unwrap());
    let delta = i128::from(frames) - end - 1;
    for value in window {
        *value = i64::try_from(i128::from(*value) + delta)
            .map_err(|_| gap("shifted context index exceeds native i64 admission"))?;
    }
    Ok(())
}
pub fn ordered_halving(value: u64) -> f64 {
    value.reverse_bits() as f64 / 18446744073709551616.
}
#[derive(Clone, Copy)]
pub struct ScheduleSettings {
    pub length: i64,
    pub overlap: i64,
    pub stride: i64,
    pub closed_loop: bool,
    pub step: u64,
}
#[derive(Clone, Copy, Debug)]
pub enum Schedule {
    UniformLooped,
    UniformStandard,
    StaticStandard,
    Batched,
}
impl Schedule {
    pub fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "looped_uniform" => Ok(Self::UniformLooped),
            "standard_uniform" => Ok(Self::UniformStandard),
            "standard_static" => Ok(Self::StaticStandard),
            "batched" => Ok(Self::Batched),
            _ => Err(error(
                "ValueError",
                format!("Unknown context_schedule '{name}'."),
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::UniformLooped => "looped_uniform",
            Self::UniformStandard => "standard_uniform",
            Self::StaticStandard => "standard_static",
            Self::Batched => "batched",
        }
    }
    pub fn windows(
        self,
        frames: i64,
        settings: ScheduleSettings,
        warning: &mut dyn FnMut(&str) -> Result<(), Error>,
    ) -> Result<Vec<Vec<i64>>, Error> {
        let mut windows = vec![];
        if frames < settings.length
            || (!matches!(self, Self::UniformLooped) && frames == settings.length)
        {
            return Ok(vec![range(0, frames.into(), 1)?]);
        }
        match self {
            Self::StaticStandard => {
                let delta = i128::from(settings.length) - i128::from(settings.overlap);
                for start in range(0, frames.into(), delta)? {
                    let end = i128::from(start) + i128::from(settings.length);
                    if end >= frames.into() {
                        let final_start = i128::from(start) - (end - i128::from(frames));
                        windows.push(range(
                            final_start,
                            final_start + i128::from(settings.length),
                            1,
                        )?);
                        break;
                    }
                    windows.push(range(start.into(), end, 1)?);
                }
            }
            Self::Batched => {
                for start in range(0, frames.into(), settings.length.into())? {
                    windows.push(range(
                        start.into(),
                        (i128::from(start) + i128::from(settings.length)).min(frames.into()),
                        1,
                    )?);
                }
            }
            Self::UniformLooped | Self::UniformStandard => {
                let ratio =
                    crate::python_numbers::integer_ratio(frames.into(), settings.length.into())?;
                if ratio == 0. {
                    warning("divide by zero encountered in log2")?;
                    return Err(error(
                        "OverflowError",
                        "cannot convert float infinity to integer",
                    ));
                }
                if ratio < 0. {
                    warning("invalid value encountered in log2")?;
                    return Err(error("ValueError", "cannot convert float NaN to integer"));
                }
                // Locked NumPy's initial-target DOUBLE_log2 uses npy_log2,
                // whose installed object directly relocates to baseline log2.
                let stride = settings.stride.min(ratio.log2().ceil() as i64 + 1);
                let fraction = ordered_halving(settings.step);
                for exponent in 0..stride {
                    let context_step = if exponent < 64 {
                        1_i64.wrapping_shl(exponent as u32)
                    } else {
                        0
                    };
                    let pad = (frames as f64 * fraction).round_ties_even() as i128;
                    let start = (fraction * context_step as f64).trunc() as i128 + pad;
                    let stop = i128::from(frames)
                        + pad
                        + if matches!(self, Self::UniformLooped) && settings.closed_loop {
                            0
                        } else {
                            -i128::from(settings.overlap)
                        };
                    let length = match settings.length.checked_mul(context_step) {
                        Some(v) => v,
                        None => {
                            warning("overflow encountered in scalar multiply")?;
                            settings.length.wrapping_mul(context_step)
                        }
                    };
                    let step = match length.checked_sub(settings.overlap) {
                        Some(v) => v,
                        None => {
                            warning("overflow encountered in scalar subtract")?;
                            length.wrapping_sub(settings.overlap)
                        }
                    };
                    for j in range(start, stop, step.into())? {
                        let end = match j.checked_add(length) {
                            Some(v) => v,
                            None => {
                                warning("overflow encountered in scalar add")?;
                                j.wrapping_add(length)
                            }
                        };
                        let window = range(j.into(), end.into(), context_step.into())?
                            .into_iter()
                            .map(|v| modulo(v.into(), frames))
                            .collect::<Result<_, _>>()?;
                        windows.push(window);
                    }
                }
                if matches!(self, Self::UniformStandard) {
                    let mut delete = vec![];
                    let mut index = 0;
                    while index < windows.len() {
                        let (roll, at) = does_roll_over(&windows[index], frames)?;
                        if roll {
                            let value = windows[index][at as usize];
                            shift_to_end(&mut windows[index], frames)?;
                            if !windows[(index + 1) % windows.len()].contains(&value) {
                                windows.insert(
                                    index + 1,
                                    range(
                                        value.into(),
                                        i128::from(value) + i128::from(settings.length),
                                        1,
                                    )?,
                                );
                            }
                        }
                        if (0..index).any(|previous| windows[index] == windows[previous]) {
                            delete.push(index);
                        }
                        index += 1;
                    }
                    for index in delete.into_iter().rev() {
                        windows.remove(index);
                    }
                }
            }
        }
        Ok(windows)
    }
}
pub fn default_windows(frames: i64) -> Result<Vec<Vec<i64>>, Error> {
    Ok(vec![range(0, frames.into(), 1)?])
}
#[derive(Clone, Copy, Debug)]
pub enum Fuse {
    Flat,
    Pyramid,
    Relative,
    OverlapLinear,
}
impl Fuse {
    pub fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "flat" => Ok(Self::Flat),
            "pyramid" => Ok(Self::Pyramid),
            "relative" => Ok(Self::Relative),
            "overlap-linear" => Ok(Self::OverlapLinear),
            _ => Err(error(
                "ValueError",
                format!("Unknown fuse_method '{name}'."),
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Pyramid => "pyramid",
            Self::Relative => "relative",
            Self::OverlapLinear => "overlap-linear",
        }
    }
    pub fn weights(
        self,
        native: &Kitchen,
        length: i64,
        frames: i64,
        indices: &[i64],
        overlap: i64,
    ) -> Result<V, Error> {
        if matches!(self, Self::Flat) {
            return Ok(V::List(
                range(0, length.into(), 1)?
                    .iter()
                    .map(|_| V::Float(1.))
                    .collect(),
            ));
        }
        if matches!(self, Self::Pyramid | Self::Relative) {
            let max = if length % 2 == 0 {
                i128::from(length).div_euclid(2)
            } else {
                (i128::from(length) + 1).div_euclid(2)
            };
            let mut weights = range(1, max + if length % 2 == 0 { 1 } else { 0 }, 1)?;
            if length % 2 != 0 {
                weights.push(max as i64);
            }
            weights.extend(range(max - if length % 2 == 0 { 0 } else { 1 }, 0, -1)?);
            return Ok(V::List(weights.into_iter().map(V::Int).collect()));
        }
        let options = (native.default_dtype()?, Device::Cpu);
        let weights = Tensor::f_ones([length], options)?;
        let minimum = indices
            .iter()
            .min()
            .ok_or_else(|| error("ValueError", "min() iterable argument is empty"))?;
        if *minimum > 0 {
            let ramp = crate::sampling::schedules::linspace(1e-37, 1., overlap, options)?;
            let target = weights.f_slice(0, Some(0), Some(overlap), 1)?;
            crate::native::index_copy_to(&target, &ramp)?;
        }
        if *indices.iter().max().unwrap() < frames - 1 {
            let ramp = crate::sampling::schedules::linspace(1., 1e-37, overlap, options)?;
            let target = weights.f_slice(0, Some(-overlap), None, 1)?;
            crate::native::index_copy_to(&target, &ramp)?;
        }
        Ok(V::tensor(weights))
    }
}
pub fn shape_for_dim(input: &Tensor, dim: i64) -> Result<Vec<i64>, Error> {
    let dims = input.size();
    let at = if dim < 0 {
        dim + i64::try_from(dims.len()).unwrap()
    } else {
        dim
    };
    let value = *dims
        .get(usize::try_from(at).unwrap_or(usize::MAX))
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let mut shape = range(0, dim.into(), 1)?
        .iter()
        .map(|_| 1)
        .collect::<Vec<_>>();
    shape.push(value);
    shape.extend(
        range(0, dims.len() as i128 - i128::from(dim) - 1, 1)?
            .iter()
            .map(|_| 1),
    );
    Ok(shape)
}
pub fn match_weights_to_dim(
    runtime: &Runtime,
    weights: &V,
    input: &Tensor,
    dim: i64,
    device: Option<Device>,
) -> Result<Tensor, Error> {
    let mut weights = match weights {
        V::Tensor(_) => unsafe {
            runtime.op(
                "aten::alias.default",
                std::slice::from_ref(weights),
                &vec![],
            )
        }?
        .as_tensor()?,
        V::List(v) | V::Tuple(v) => {
            let values = v
                .iter()
                .map(|v| crate::python_numbers::float_value(runtime, v))
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::f_from_slice(&values)?.f_to_kind(runtime.native.default_dtype()?)?
        }
        _ => {
            return Err(gap(
                "non-list/non-tensor legacy weight constructor requires admission",
            ));
        }
    };
    if let Some(device) = device {
        weights = weights.f_to_device(device)?;
    }
    for _ in 0..dim {
        weights = weights.f_unsqueeze(0)?;
    }
    for _ in 0..(input.dim() as i128 - i128::from(dim) - 1) {
        weights = weights.f_unsqueeze(-1)?;
    }
    Ok(weights)
}
pub fn apply_freenoise(
    native: &Kitchen,
    noise: &V,
    dim: i64,
    length: i64,
    overlap: i64,
    seed: u64,
    info: &mut dyn FnMut(&str) -> Result<(), Error>,
) -> Result<V, Error> {
    apply_freenoise_value(native, noise, dim, length, overlap, &V::UInt(seed), info)
}
fn apply_freenoise_value(
    native: &Kitchen,
    noise: &V,
    dim: i64,
    length: i64,
    overlap: i64,
    seed: &V,
    info: &mut dyn FnMut(&str) -> Result<(), Error>,
) -> Result<V, Error> {
    info("Context windows: Applying FreeNoise")?;
    let seed = match seed {
        V::Int(v) => *v as u64,
        V::UInt(v) => *v,
        value => {
            return Err(error(
                "RuntimeError",
                format!(
                    "manual_seed expected a long, but got {}",
                    crate::backend_triton_fp8::typename(value)
                ),
            ));
        }
    };
    let mut generator = native.generator(Device::Cpu, seed)?;
    let tensor = noise.as_tensor()?;
    let shape = tensor.size();
    let at = if dim < 0 {
        dim + shape.len() as i64
    } else {
        dim
    };
    let frames = *shape
        .get(usize::try_from(at).unwrap_or(usize::MAX))
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let delta = i128::from(length) - i128::from(overlap);
    for start in range(0, i128::from(frames) - i128::from(length), delta)? {
        let place = i128::from(start) + i128::from(length);
        let actual = delta.min(i128::from(frames) - place);
        if actual <= 0 {
            break;
        }
        let actual =
            i64::try_from(actual).map_err(|_| gap("FreeNoise delta exceeds native i64"))?;
        let indices = generator.randperm(actual)?.f_add_scalar(start)?;
        let mut index = std::iter::repeat_with(|| None)
            .take(at as usize)
            .collect::<Vec<Option<&Tensor>>>();
        index.push(Some(&indices));
        let source = tensor.f_index(index.as_slice())?;
        let target = tensor.f_slice(
            at,
            Some(place as i64),
            Some((place + i128::from(actual)) as i64),
            1,
        )?;
        crate::native::index_copy_to(&target, &source)?;
    }
    Ok(noise.clone())
}
