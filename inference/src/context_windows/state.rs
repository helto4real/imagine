//! Per-modality context state and source-owned guide/window containers.
use super::{
    GuideOverlap, IndexWindow, LatentShape, LatentShapes, WindowHandle, WindowList,
    compute_guide_overlap, error, gap,
};
use crate::{
    conditioning::{Condition, Kind},
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Tensor};

pub type ModalityIndices = WindowList<WindowList<i64>>;
pub trait ModalityMapper {
    fn class_name(&self) -> &str;
    fn map_context_window_to_modalities(
        &mut self,
        indices: WindowList<i64>,
        shapes: LatentShapes,
        dim: i64,
    ) -> Result<ModalityIndices, Error>;
}
pub struct WindowingState {
    pub latents: WindowList<V>,
    pub guide_latents: WindowList<Option<V>>,
    pub guide_entries: WindowList<Option<Options>>,
    pub keyframe_idxs: WindowList<Option<V>>,
    pub latent_shapes: Option<LatentShapes>,
    pub dim: i64,
    pub is_multimodal: bool,
    pub temporal_downscale_ratio: i64,
}
fn at<T: Clone>(list: &WindowList<T>, index: usize) -> Result<T, Error> {
    list.borrow()
        .get(index)
        .cloned()
        .ok_or_else(|| error("IndexError", "list index out of range"))
}
pub(super) fn shape_at(shape: &[i64], dim: i64) -> Result<i64, Error> {
    let index = if dim < 0 {
        shape.len() as i128 + i128::from(dim)
    } else {
        i128::from(dim)
    };
    usize::try_from(index)
        .ok()
        .and_then(|i| shape.get(i))
        .copied()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))
}
fn dict_field(entry: &Options, key: &str) -> Result<Options, Error> {
    entry
        .get(key)?
        .ok_or_else(|| error("KeyError", format!("'{key}'")))
}
fn sequence(value: &Options) -> Result<Vec<Options>, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().clone()),
        Options::Tuple(v) => Ok((**v).clone()),
        _ => Err(gap("window state requires source sequence binding")),
    }
}
fn integral(value: &Options) -> Result<i64, Error> {
    match value.snapshot()? {
        V::Int(v) => Ok(v),
        V::Bool(v) => Ok(i64::from(v)),
        V::UInt(v) => i64::try_from(v).map_err(|_| gap("window state metadata exceeds native i64")),
        _ => Err(gap(
            "window state metadata requires original integral binding",
        )),
    }
}
impl WindowingState {
    pub fn prepare_window(
        &self,
        window: &WindowHandle,
        model: &mut dyn ModalityMapper,
    ) -> Result<WindowHandle, Error> {
        if !self.is_multimodal {
            return Ok(window.clone());
        }
        let x = at(&self.latents, 0)?.as_tensor()?;
        let shapes = self
            .latent_shapes
            .as_ref()
            .ok_or_else(|| error("TypeError", "'NoneType' object is not subscriptable"))?;
        let primary = shapes.get(0)?.dimension(self.dim)?;
        let overlap = window.borrow().overlap;
        let map_shapes = if native::context_size(&x, self.dim)? != primary {
            let list = shapes.entries()?;
            let mut video = shapes.get(0)?.values()?;
            let index = if self.dim < 0 {
                (video.len() as i64 + self.dim) as usize
            } else {
                self.dim as usize
            };
            video[index] = native::context_size(&x, self.dim)?;
            let mut copy = list;
            copy[0] = LatentShape::Size(Rc::new(video));
            LatentShapes::new(copy)
        } else {
            shapes.clone()
        };
        let indices = window.borrow().indices.clone();
        let mapped = match model.map_context_window_to_modalities(indices, map_shapes, self.dim) {
            Err(caught) if caught.is_reference("AttributeError") => {
                return Err(error(
                    "NotImplementedError",
                    format!(
                        "{} must implement map_context_window_to_modalities for multimodal context windows.",
                        model.class_name()
                    ),
                ));
            }
            other => other?,
        };
        let modalities = Rc::new(RefCell::new(vec![]));
        for i in 1..self.latents.borrow().len() {
            let total = shape_at(&at(&self.latents, i)?.as_tensor()?.size(), self.dim)?;
            let ratio = if primary > 0 {
                crate::python_numbers::integer_ratio(total.into(), primary.into())?
            } else {
                1.
            };
            let mod_overlap = (overlap as f64 * ratio).round_ties_even().max(0.);
            if mod_overlap >= 9223372036854775808. {
                return Err(gap("modality overlap exceeds native i64"));
            }
            let child =
                IndexWindow::new(at(&mapped, i)?, self.dim, total, None, mod_overlap as i64)?;
            modalities.borrow_mut().push((i as i64, child));
        }
        IndexWindow::new(
            window.borrow().indices.clone(),
            self.dim,
            shape_at(&x.size(), self.dim)?,
            Some(modalities),
            overlap,
        )
    }
    pub fn slice_for_window(
        &self,
        window: &WindowHandle,
        retain: &[i64],
        device: Option<Device>,
    ) -> Result<(Vec<V>, Vec<i64>), Error> {
        let (mut sliced, mut counts) = (vec![], vec![]);
        for i in 0..self.latents.borrow().len() {
            let modality = IndexWindow::modality(window, i as i64)?;
            let value = modality.borrow().get_tensor(
                &at(&self.latents, i)?,
                device,
                None,
                if i == 0 { retain } else { &[] },
            )?;
            let (value, count) = if at(&self.guide_entries, i)?.is_some() {
                self.inject_guide_frames(&value, &modality, i)?
            } else {
                (value, 0)
            };
            sliced.push(value);
            counts.push(count);
        }
        Ok((sliced, counts))
    }
    pub fn strip_guide_frames(
        &self,
        outputs: &WindowList<WindowList<V>>,
        counts: &[i64],
        window: &WindowHandle,
    ) -> Result<(), Error> {
        for i in 0..self.latents.borrow().len() {
            if *counts
                .get(i)
                .ok_or_else(|| error("IndexError", "list index out of range"))?
                > 0
            {
                let length = IndexWindow::modality(window, i as i64)?
                    .borrow()
                    .indices
                    .borrow()
                    .len() as i64;
                for ci in 0..outputs.borrow().len() {
                    let list = at(outputs, ci)?;
                    let t = at(&list, i)?.as_tensor()?;
                    list.borrow_mut()[i] = V::tensor(t.f_narrow(self.dim, 0, length)?);
                }
            }
        }
        Ok(())
    }
    pub fn inject_guide_frames(
        &self,
        latent: &V,
        window: &WindowHandle,
        i: usize,
    ) -> Result<(V, i64), Error> {
        let entries = at(&self.guide_entries, i)?
            .ok_or_else(|| error("TypeError", "'NoneType' object is not iterable"))?;
        let frames = at(&self.guide_latents, i)?;
        let keyframes = at(&self.keyframe_idxs, i)?;
        let entries = sequence(&entries)?;
        let overlap = if entries.is_empty() {
            GuideOverlap {
                suffix_indices: vec![],
                overlap_info: vec![],
                keyframe_local_positions: vec![],
            }
        } else {
            let keyframes = keyframes
                .ok_or_else(|| error("TypeError", "'NoneType' object is not subscriptable"))?
                .as_tensor()?;
            compute_guide_overlap(
                &entries,
                &keyframes,
                self.temporal_downscale_ratio,
                &window.borrow().indices.borrow(),
            )?
        };
        let count = overlap.count() as i64;
        let mut positions = overlap.keyframe_local_positions;
        if window.borrow().causal_anchor.is_some_and(|v| v >= 0) {
            for p in &mut positions {
                *p += 1;
            }
        }
        window.borrow_mut().guide_frames_indices =
            Rc::new(RefCell::new(overlap.suffix_indices.clone()));
        window.borrow_mut().guide_overlap_info =
            Rc::new(RefCell::new(overlap.overlap_info.clone()));
        window.borrow_mut().guide_kf_local_positions = Rc::new(RefCell::new(positions));
        let mut factors = vec![];
        if count > 0 {
            let frames = frames
                .as_ref()
                .ok_or_else(|| {
                    error(
                        "AttributeError",
                        "'NoneType' object has no attribute 'shape'",
                    )
                })?
                .as_tensor()?;
            let height = shape_at(&frames.size(), 3)?;
            for (entry_idx, _) in overlap.overlap_info {
                let shape = sequence(&dict_field(&entries[entry_idx as usize], "latent_shape")?)?;
                let entry_height = integral(
                    shape
                        .get(1)
                        .ok_or_else(|| error("IndexError", "list index out of range"))?,
                )?;
                if entry_height == 0 {
                    return Err(error(
                        "ZeroDivisionError",
                        "integer division or modulo by zero",
                    ));
                }
                let n = i128::from(height);
                let d = i128::from(entry_height);
                let r = n % d;
                factors.push(
                    i64::try_from(n / d - i128::from(r != 0 && (r < 0) != (d < 0)))
                        .map_err(|_| gap("guide downscale exceeds native i64"))?,
                );
            }
        }
        window.borrow_mut().guide_downscale_factors = Rc::new(RefCell::new(factors));
        if count > 0 {
            let frames = frames.unwrap().as_tensor()?;
            let selected = native::context_index(&frames, &overlap.suffix_indices, self.dim)?;
            return Ok((
                V::tensor(Tensor::f_cat(&[latent.as_tensor()?, selected], self.dim)?),
                count,
            ));
        }
        Ok((latent.clone(), 0))
    }
    pub fn patch_latent_shapes(&self, conds: &Options, new_shapes: &Options) -> Result<(), Error> {
        if !self.is_multimodal {
            return Ok(());
        }
        for list in sequence(conds)? {
            if matches!(list, Options::Atom(V::None)) {
                continue;
            }
            for entry in sequence(&list)? {
                let model_conds = entry
                    .get("model_conds")?
                    .unwrap_or_else(|| Options::from_value(V::Dict(vec![])));
                if model_conds.get("latent_shapes")?.is_some() {
                    model_conds.set(
                        "latent_shapes",
                        Options::Object(Rc::new(Condition::from_options(
                            Kind::Constant,
                            new_shapes.clone(),
                        ))),
                    )?;
                }
            }
        }
        Ok(())
    }
}
