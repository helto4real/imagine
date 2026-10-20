//! Handler FreeNoise preprocessing, including guide stripping and per-modality
//! context scaling. The primitive keeps its original private CPU generator.
use super::{
    ConditionAttributes, Handler, LatentShapes, apply_freenoise_value, error, gap, pack_latents,
    unpack_latents,
};
use crate::{
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
fn scaled(value: i64, ratio: f64, minimum: i64) -> Result<i64, Error> {
    let n = (value as f64 * ratio).round_ties_even();
    if n.is_nan() {
        return Err(error("ValueError", "cannot convert float NaN to integer"));
    }
    if !n.is_finite() {
        return Err(error(
            "OverflowError",
            "cannot convert float infinity to integer",
        ));
    }
    if n >= 9223372036854775808. {
        return Err(gap("context FreeNoise scaled length exceeds i64"));
    }
    Ok((n as i64).max(minimum))
}
impl Handler {
    pub fn apply_freenoise(
        &self,
        native: &native::Kitchen,
        noise: &V,
        conditions: &Options,
        seed: u64,
        attributes: &dyn ConditionAttributes,
        info: &mut dyn FnMut(&str) -> Result<(), Error>,
    ) -> Result<V, Error> {
        self.apply_freenoise_value(native, noise, conditions, &V::UInt(seed), attributes, info)
    }
    pub fn apply_freenoise_value(
        &self,
        native: &native::Kitchen,
        noise: &V,
        conditions: &Options,
        seed: &V,
        attributes: &dyn ConditionAttributes,
        info: &mut dyn FnMut(&str) -> Result<(), Error>,
    ) -> Result<V, Error> {
        let entries = Self::get_guide_entries(conditions, attributes)?;
        let guide_count = super::metadata::guide_count(&entries)?;
        let source_shapes = Self::get_latent_shapes(conditions, attributes)?;
        let shapes = if matches!(source_shapes, Options::Atom(V::None)) {
            None
        } else {
            Some(LatentShapes::from_options(&source_shapes)?)
        };
        if let Some(shapes) = shapes
            && shapes.len()? > 1
        {
            let mut modalities = unpack_latents(noise, &shapes)?;
            let dim = self.settings.borrow().dim;
            let primary_total = shapes.get(0)?.dimension(dim)?;
            let primary_video = native::context_size(&modalities[0].as_tensor()?, dim)?
                .checked_sub(guide_count)
                .ok_or_else(|| gap("context FreeNoise video count exceeds i64"))?;
            let part = V::tensor(modalities[0].as_tensor()?.f_narrow(dim, 0, primary_video)?);
            let (dim, length, overlap) = {
                let s = self.settings.borrow();
                (s.dim, s.length, s.overlap)
            };
            apply_freenoise_value(native, &part, dim, length, overlap, seed, info)?;
            for (i, modality) in modalities.iter_mut().enumerate().skip(1) {
                let dim = self.settings.borrow().dim;
                let total = shapes.get(i)?.dimension(dim)?;
                let ratio = if primary_total > 0 {
                    crate::python_numbers::integer_ratio(total.into(), primary_total.into())?
                } else {
                    1.
                };
                let length = scaled(self.settings.borrow().length, ratio, 1)?;
                let overlap = scaled(self.settings.borrow().overlap, ratio, 0)?;
                let dim = self.settings.borrow().dim;
                *modality =
                    apply_freenoise_value(native, modality, dim, length, overlap, seed, info)?;
            }
            return pack_latents(&modalities).map(|(noise, _)| noise);
        }
        let dim = self.settings.borrow().dim;
        let video = native::context_size(&noise.as_tensor()?, dim)?
            .checked_sub(guide_count)
            .ok_or_else(|| gap("context FreeNoise video count exceeds i64"))?;
        let part = V::tensor(noise.as_tensor()?.f_narrow(dim, 0, video)?);
        let (dim, length, overlap) = {
            let s = self.settings.borrow();
            (s.dim, s.length, s.overlap)
        };
        apply_freenoise_value(native, &part, dim, length, overlap, seed, info)?;
        Ok(noise.clone())
    }
}
