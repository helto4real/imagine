//! Original packed multimodal latent reshape/cat/slice operations.
use super::{LatentShape, LatentShapes, error, gap};
use crate::{
    execution_value::Value as V,
    native::{self, Error},
};
use std::rc::Rc;
use tch::Tensor;

pub fn pack_latents(latents: &[V]) -> Result<(V, LatentShapes), Error> {
    let (mut shapes, mut tensors) = (vec![], vec![]);
    for value in latents {
        let tensor = value.as_tensor()?;
        let shape = tensor.size();
        shapes.push(LatentShape::Size(Rc::new(shape.clone())));
        let batch = *shape
            .first()
            .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
        tensors.push(tensor.f_reshape([batch, 1, -1])?);
    }
    let latent = Tensor::f_cat(&tensors, -1)?;
    Ok((V::tensor(latent), LatentShapes::new(shapes)))
}
pub fn unpack_latents(combined: &V, shapes: &LatentShapes) -> Result<Vec<V>, Error> {
    if shapes.len()? <= 1 {
        return Ok(vec![combined.clone()]);
    }
    let mut output = vec![];
    let mut combined = combined.as_tensor()?;
    let mut index = 0;
    while let Some(shape) = shapes.next(index)? {
        index += 1;
        let shape = shape.values()?;
        let cut = shape.iter().skip(1).try_fold(1i128, |v, n| {
            v.checked_mul(i128::from(*n))
                .ok_or_else(|| gap("unpacked shape product exceeds native i128"))
        })?;
        let cut = i64::try_from(cut).map_err(|_| gap("unpacked shape slice exceeds native i64"))?;
        let tens = native::context_slice(&combined, 2, 0, cut)?;
        combined = native::context_slice(&combined, 2, cut, i64::MAX)?;
        let batch = *tens
            .size()
            .first()
            .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
        let mut target = vec![batch];
        target.extend(shape.into_iter().skip(1));
        output.push(V::tensor(tens.f_reshape(target.as_slice())?));
    }
    Ok(output)
}
