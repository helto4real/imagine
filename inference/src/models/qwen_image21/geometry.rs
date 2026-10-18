use super::zero_centered_rms_norm;
use crate::{execution::Runtime, execution_value::Value, native::Error};
use tch::{Device, Kind, Tensor};

pub struct TextProjection {
    pub norm_weight: Tensor,
    pub in_weight: Value,
    pub out_weight: Value,
    pub eps: f64,
}
impl TextProjection {
    /// # Safety
    /// Resident weights and inputs must satisfy the selected original linear contract.
    pub unsafe fn forward(&self, rt: &Runtime, x: &Tensor) -> Result<Tensor, Error> {
        let x = zero_centered_rms_norm(x, &self.norm_weight, self.eps)?;
        let x = unsafe { linear(rt, x, &self.in_weight)? }.f_gelu("tanh")?;
        unsafe { linear(rt, x, &self.out_weight) }
    }
}
pub(super) unsafe fn linear(rt: &Runtime, x: Tensor, weight: &Value) -> Result<Tensor, Error> {
    // F.linear decomposes a non-contiguous rank-3 input into t/expand/bmm.
    // Kitchen has no INT8 expand handler: its original dispatch dequantizes
    // the transposed weight before bmm. Forcing a contiguous input here would
    // select INT8 mm instead and change the result.
    if x.dim() == 3 && !x.is_contiguous() && matches!(weight, Value::Quantized(_)) {
        let transposed =
            unsafe { rt.op("aten::t.default", std::slice::from_ref(weight), &vec![])? };
        let w = unsafe { rt.dequantize_args(&transposed)? }.as_tensor()?;
        let shape = x.size();
        let wshape = w.size();
        let left = x
            .f_expand([shape[0], shape[1], shape[2]], false)?
            .f_view([shape[0], shape[1], shape[2]])?;
        let right = w
            .f_expand([shape[0], wshape[0], wshape[1]], false)?
            .f_view([shape[0], wshape[0], wshape[1]])?;
        return Ok(left
            .f_bmm(&right)?
            .f_view([shape[0], shape[1], wshape[1]])?);
    }
    unsafe {
        rt.op(
            "aten::linear.default",
            &[Value::tensor(x), weight.clone(), Value::None],
            &vec![],
        )?
        .as_tensor()
    }
}

pub struct PositionEncoding {
    pub axes: [i64; 3],
    pub theta: f64,
    pub supports_fp64: bool,
}
impl Default for PositionEncoding {
    fn default() -> Self {
        Self {
            axes: [16, 56, 56],
            theta: 10000.,
            supports_fp64: true,
        }
    }
}
impl PositionEncoding {
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor, Error> {
        let mut values = vec![];
        for axis in 0..ids.size()[ids.dim() - 1] {
            let dim = self.axes[axis as usize];
            if dim % 2 != 0 {
                return Err(Error::ReferenceException {
                    class: "AssertionError",
                    message: String::new(),
                });
            }
            let device = if self.supports_fp64 {
                ids.device()
            } else {
                Device::Cpu
            };
            let scale = Tensor::f_linspace(
                0.,
                (dim - 2) as f64 / dim as f64,
                dim / 2,
                (Kind::Double, device),
            )?;
            let omega = Tensor::f_pow_scalar(self.theta, &scale)?
                .f_reciprocal()?
                .f_mul_scalar(1.0)?;
            let pos = ids
                .f_select(-1, axis)?
                .f_to_device_(device, Kind::Float, false, false)?;
            let out = Tensor::f_einsum("...n,d->...nd", &[pos, omega], None::<&[i64]>)?;
            let shape = out.size();
            let rope = Tensor::f_stack(
                &[
                    out.f_cos()?,
                    out.f_sin()?.f_neg()?,
                    out.f_sin()?,
                    out.f_cos()?,
                ],
                -1,
            )?
            .f_reshape([shape[0], shape[1], shape[2], 2, 2])?
            .f_to_device_(ids.device(), Kind::Float, false, false)?;
            values.push(rope);
        }
        Ok(Tensor::f_cat(&values, -3)?.f_unsqueeze(1)?)
    }
}

pub struct Segment {
    pub start: i64,
    pub end: i64,
    pub mask: Option<Tensor>,
}
pub struct Sequence {
    pub hidden: Tensor,
    pub position_encoding: Tensor,
    pub segments: Vec<Segment>,
}
pub struct SequenceProjections<'a> {
    pub text: &'a TextProjection,
    pub image: &'a Value,
}

/// # Safety
/// All resident projection weights and inputs must satisfy original kernel contracts.
pub unsafe fn build_sequence(
    rt: &Runtime,
    x: &Tensor,
    context: &Tensor,
    refs: &[Tensor],
    image_slots: &[i64],
    projections: &SequenceProjections<'_>,
    pe: &PositionEncoding,
) -> Result<Sequence, Error> {
    let txt = unsafe { projections.text.forward(rt, context)? };
    assemble_sequence(x, &txt, refs, image_slots, pe, |image| unsafe {
        linear(rt, image.shallow_clone(), projections.image)
    })
}

pub(super) fn assemble_sequence(
    x: &Tensor,
    txt: &Tensor,
    refs: &[Tensor],
    image_slots: &[i64],
    pe: &PositionEncoding,
    mut image: impl FnMut(&Tensor) -> Result<Tensor, Error>,
) -> Result<Sequence, Error> {
    let mut slots = image_slots.to_vec();
    slots.resize(slots.len() + refs.len(), txt.size()[1]);
    slots.truncate(refs.len());
    let mut bounds = vec![0];
    bounds.extend(slots);
    bounds.push(txt.size()[1]);
    let mut parts = vec![];
    let mut ids = vec![];
    let mut segments = vec![];
    let mut pos = 0;
    let mut length = 0;
    let images = refs.iter().chain(std::iter::once(x));
    for (pair, img) in bounds.windows(2).zip(images) {
        let n = pair[1] - pair[0];
        if n > 0 {
            parts.push(txt.f_slice(1, pair[0], pair[1], 1)?);
            ids.push(
                Tensor::f_arange_start(pos, pos + n, (Kind::Float, x.device()))?
                    .f_unsqueeze(1)?
                    .f_expand([n, 3], false)?,
            );
            segments.push(Segment {
                start: length,
                end: length + n,
                mask: Some(
                    Tensor::f_ones([n, length + n], (Kind::Bool, x.device()))?.f_tril(length)?,
                ),
            });
            pos += n;
            length += n;
        }
        let shape = img.size();
        let h = shape[shape.len() - 2];
        let w = shape[shape.len() - 1];
        parts.push(image(&img.f_flatten(2, -1)?.f_transpose(1, 2)?)?);
        let target = x.size();
        let hh = Tensor::f_arange(h, (Kind::Float, x.device()))?
            .f_sub_scalar(h - h / 2)?
            .f_add_scalar(0.5 * (h % 2 - target[target.len() - 2] % 2) as f64)?;
        let ww = Tensor::f_arange(w, (Kind::Float, x.device()))?
            .f_sub_scalar(w - w / 2)?
            .f_add_scalar(0.5 * (w % 2 - target[target.len() - 1] % 2) as f64)?;
        ids.push(
            Tensor::f_stack(
                &[
                    Tensor::f_full([h, w], pos, (Kind::Float, x.device()))?,
                    hh.f_unsqueeze(1)?.f_expand([h, w], false)?,
                    ww.f_unsqueeze(0)?.f_expand([h, w], false)?,
                ],
                -1,
            )?
            .f_flatten(0, 1)?,
        );
        segments.push(Segment {
            start: length,
            end: length + h * w,
            mask: None,
        });
        pos += h.max(w);
        length += h * w;
    }
    let position_encoding = pe
        .forward(&Tensor::f_cat(&ids, 0)?.f_unsqueeze(0)?)?
        .f_transpose(1, 2)?
        .f_contiguous()?;
    Ok(Sequence {
        hidden: Tensor::f_cat(&parts, 1)?,
        position_encoding,
        segments,
    })
}
