//! Original state construction. Model metadata is read only after conditioning
//! extraction and tensor slicing, with the original local owners still live.
use super::{
    ConditionAttributes, Handler, LatentShapes, WindowList, WindowingState, gap, unpack_latents,
};
use crate::{
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
use std::{cell::RefCell, rc::Rc};
pub struct WindowBuildLocals<'a> {
    pub is_multimodal: bool,
    pub source_shapes: &'a Options,
    pub unpacked: &'a [V],
    pub latents: &'a WindowList<V>,
    pub guide_latents: &'a WindowList<Option<V>>,
    pub guide_entries: &'a WindowList<Option<Options>>,
    pub keyframe_idxs: &'a WindowList<Option<V>>,
    pub extracted_guides: &'a Options,
    pub extracted_keyframes: &'a Options,
    pub primary: Option<&'a V>,
}
pub trait WindowStateModel {
    fn temporal_downscale_ratio(&self) -> Result<i64, Error>;
    fn source_locals(&self, _: &WindowBuildLocals<'_>) -> Result<(), Error> {
        Ok(())
    }
}
fn none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
impl Handler {
    pub fn should_use_context(
        &self,
        x: &V,
        conditions: &Options,
        attributes: &dyn ConditionAttributes,
        model: &dyn WindowStateModel,
        info: &mut dyn FnMut(&str) -> Result<(), Error>,
    ) -> Result<bool, Error> {
        let state = self.build_window_state(x, conditions, attributes, model)?;
        let frames = {
            let latents = state.latents.borrow();
            native::context_size(&latents[0].as_tensor()?, self.settings.borrow().dim)?
        };
        let use_context = frames > self.settings.borrow().length;
        if use_context {
            let message = {
                let s = self.settings.borrow();
                format!(
                    "\nUsing context windows: Context length {} with overlap {} for {frames} frames.",
                    s.length, s.overlap
                )
            };
            info(&message)?;
            let retain = self.settings.borrow().condition_retain.clone();
            let log_retain = !retain.borrow().is_empty();
            if log_retain {
                let message = format!("Retaining original cond for indexes: {:?}", retain.borrow());
                info(&message)?;
            }
            let retain = self.settings.borrow().latent_retain.clone();
            let log_retain = !retain.borrow().is_empty();
            if log_retain {
                let message = format!(
                    "Retaining original latent for indexes: {:?}",
                    retain.borrow()
                );
                info(&message)?;
            }
            return Ok(true);
        }
        let message = format!(
            "\nNot using context windows since context length ({}) exceeds input frames ({frames}).",
            self.settings.borrow().length
        );
        info(&message)?;
        Ok(false)
    }
    pub fn build_window_state(
        &self,
        x: &V,
        conditions: &Options,
        attributes: &dyn ConditionAttributes,
        model: &dyn WindowStateModel,
    ) -> Result<WindowingState, Error> {
        let source_shapes = Self::get_latent_shapes(conditions, attributes)?;
        let shapes = if none(&source_shapes) {
            None
        } else {
            Some(LatentShapes::from_options(&source_shapes)?)
        };
        let multimodal = shapes
            .as_ref()
            .map(|v| v.len().map(|n| n > 1))
            .transpose()?
            .unwrap_or(false);
        let unpacked = if multimodal {
            unpack_latents(x, shapes.as_ref().unwrap())?
        } else {
            vec![x.clone()]
        };
        let latents = Rc::new(RefCell::new(unpacked.clone()));
        let guides = Rc::new(RefCell::new(vec![None; unpacked.len()]));
        let entries = Rc::new(RefCell::new(vec![None; unpacked.len()]));
        let keyframes = Rc::new(RefCell::new(vec![None; unpacked.len()]));
        let extracted_guides = Self::get_guide_entries(conditions, attributes)?;
        let extracted_keyframes = Self::get_keyframe_indices(conditions, attributes)?;
        let mut primary = None;
        if !none(&extracted_guides) {
            let count = super::metadata::guide_count(&extracted_guides)?;
            if count > 0 {
                primary = Some(unpacked[0].clone());
                let raw = primary.as_ref().unwrap().as_tensor()?;
                let dim = self.settings.borrow().dim;
                let latent_count = native::context_size(&raw, dim)?
                    .checked_sub(count)
                    .ok_or_else(|| gap("context state latent count exceeds i64"))?;
                latents.borrow_mut()[0] =
                    V::tensor(raw.f_narrow(self.settings.borrow().dim, 0, latent_count)?);
                guides.borrow_mut()[0] = Some(V::tensor(raw.f_narrow(
                    self.settings.borrow().dim,
                    latent_count,
                    count,
                )?));
                entries.borrow_mut()[0] = Some(extracted_guides.clone());
                keyframes.borrow_mut()[0] = match &extracted_keyframes {
                    Options::Atom(V::None) => None,
                    Options::Atom(v) => Some(v.clone()),
                    _ => return Err(gap("context keyframes require their source value binding")),
                };
            }
        }
        let dim = self.settings.borrow().dim;
        model.source_locals(&WindowBuildLocals {
            is_multimodal: multimodal,
            source_shapes: &source_shapes,
            unpacked: &unpacked,
            latents: &latents,
            guide_latents: &guides,
            guide_entries: &entries,
            keyframe_idxs: &keyframes,
            extracted_guides: &extracted_guides,
            extracted_keyframes: &extracted_keyframes,
            primary: primary.as_ref(),
        })?;
        let ratio = model.temporal_downscale_ratio()?;
        Ok(WindowingState {
            latents,
            guide_latents: guides,
            guide_entries: entries,
            keyframe_idxs: keyframes,
            latent_shapes: shapes,
            dim,
            is_multimodal: multimodal,
            temporal_downscale_ratio: ratio,
        })
    }
}
