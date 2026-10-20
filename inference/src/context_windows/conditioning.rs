//! Source temporal conditioning and concatenated guide overlap bookkeeping.
use super::{IndexWindow, error, gap};
use crate::{
    conditioning::Condition,
    execution_value::Value as V,
    native::{self, Error},
    patching::options::Options,
};
use tch::{Device, Tensor};

/// Built-in COND classes all inherit the original `_copy_with` constructor.
#[allow(clippy::too_many_arguments)]
pub fn slice_condition(
    cond: &Condition,
    window: &IndexWindow,
    x: &Tensor,
    device: Option<Device>,
    temporal_dim: i64,
    temporal_scale: i64,
    temporal_offset: i64,
    retain: &[i64],
) -> Result<Option<Condition>, Error> {
    let value = cond.value()?;
    if matches!(value, V::Quantized(_)) {
        return Err(gap(
            "quantized temporal conditioning indexing requires its source layout binding",
        ));
    }
    if !matches!(value, V::Tensor(_)) {
        return Ok(None);
    }
    let tensor = value.as_tensor()?;
    if temporal_dim >= tensor.dim() as i64 {
        return Ok(None);
    }
    let size = native::context_size(&tensor, temporal_dim)?;
    if temporal_scale == 1 {
        let expected =
            i128::from(native::context_size(x, window.dim)?) - i128::from(temporal_offset);
        if i128::from(size) != expected {
            return Ok(None);
        }
    }
    if temporal_offset == 0 && temporal_scale == 1 {
        return Ok(Some(cond.copy_with(window.get_tensor(
            &value,
            device,
            Some(temporal_dim),
            retain,
        )?)));
    }
    let indices = window.indices.borrow();
    let mut selected = if temporal_offset > 0 {
        let skip = temporal_offset - i64::from(window.causal_anchor.is_some_and(|v| v >= 0));
        indices
            .iter()
            .skip(skip as usize)
            .map(|i| i128::from(*i) - i128::from(temporal_offset))
            .filter(|v| *v >= 0)
            .collect::<Vec<_>>()
    } else {
        indices.iter().map(|i| i128::from(*i)).collect()
    };
    if selected.is_empty() {
        return Ok(None);
    }
    if temporal_scale > 1 {
        let mut scaled = vec![];
        for i in selected {
            for k in 0..temporal_scale {
                let si = i * i128::from(temporal_scale) + i128::from(k);
                if si < i128::from(size) {
                    scaled.push(si);
                }
            }
        }
        selected = scaled;
        if selected.is_empty() {
            return Ok(None);
        }
    }
    let selected = selected
        .into_iter()
        .map(|v| {
            i64::try_from(v).map_err(|_| gap("temporal conditioning index exceeds native i64"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sliced = native::context_index(&tensor, &selected, temporal_dim)?;
    let sliced = if let Some(device) = device {
        sliced.f_to_device(device)?
    } else {
        sliced
    };
    Ok(Some(cond.copy_with(V::tensor(sliced))))
}

pub struct GuideOverlap {
    pub suffix_indices: Vec<i64>,
    pub overlap_info: Vec<(i64, i64)>,
    pub keyframe_local_positions: Vec<i64>,
}
impl GuideOverlap {
    pub fn count(&self) -> usize {
        self.suffix_indices.len()
    }
}
fn required(entry: &Options, key: &str) -> Result<Options, Error> {
    entry
        .get(key)?
        .ok_or_else(|| error("KeyError", format!("'{key}'")))
}
fn integer(value: &V) -> Result<i64, Error> {
    match value {
        V::Int(v) => Ok(*v),
        V::Bool(v) => Ok(i64::from(*v)),
        V::UInt(v) => i64::try_from(*v).map_err(|_| gap("guide integer exceeds native i64")),
        V::Float(v) if !v.is_finite() => Err(error(
            if v.is_nan() {
                "ValueError"
            } else {
                "OverflowError"
            },
            if v.is_nan() {
                "cannot convert float NaN to integer"
            } else {
                "cannot convert float infinity to integer"
            },
        )),
        V::Float(v) if *v >= i64::MIN as f64 && *v < 9223372036854775808. => Ok(v.trunc() as i64),
        _ => Err(gap(
            "guide integer requires original integral value admission",
        )),
    }
}
fn range_integer(value: &V) -> Result<i64, Error> {
    match value {
        V::Int(_) | V::UInt(_) | V::Bool(_) => integer(value),
        _ => Err(error(
            "TypeError",
            format!(
                "'{}' object cannot be interpreted as an integer",
                crate::backend_triton_fp8::typename(value)
            ),
        )),
    }
}
pub fn compute_guide_overlap(
    entries: &[Options],
    keyframe_idxs: &Tensor,
    ratio: i64,
    window: &[i64],
) -> Result<GuideOverlap, Error> {
    let mut out = GuideOverlap {
        suffix_indices: vec![],
        overlap_info: vec![],
        keyframe_local_positions: vec![],
    };
    let (mut suffix_base, mut token_offset) = (0i128, 0i128);
    let mut floating_token_offset = false;
    for (entry_idx, entry) in entries.iter().enumerate() {
        if floating_token_offset {
            return Err(error(
                "IndexError",
                "only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got float)",
            ));
        }
        let token = i64::try_from(token_offset)
            .map_err(|_| gap("guide token offset exceeds native i64"))?;
        let first = native::context_scalar_index(keyframe_idxs, &[0, 0, token, 0])?;
        // .item() retains the original scalar type before Python int truncation.
        if first.is_complex() {
            return Err(error(
                "TypeError",
                "int() argument must be a string, a bytes-like object or a real number, not 'complex'",
            ));
        }
        let pixel = if first.is_floating_point() {
            integer(&V::Float(first.f_double_value(&[])?))?
        } else {
            first.f_int64_value(&[])?
        };
        if ratio == 0 {
            return Err(error(
                "ZeroDivisionError",
                "integer division or modulo by zero",
            ));
        }
        let numerator = i128::from(pixel) + i128::from(ratio) - 1;
        let divisor = i128::from(ratio);
        let remainder = numerator % divisor;
        let start =
            numerator / divisor - i128::from(remainder != 0 && (remainder < 0) != (divisor < 0));
        let shape = required(entry, "latent_shape")?.snapshot()?;
        let length = match shape {
            V::List(v) => range_integer(
                v.first()
                    .ok_or_else(|| error("IndexError", "list index out of range"))?,
            )?,
            V::Tuple(v) => range_integer(
                v.first()
                    .ok_or_else(|| error("IndexError", "tuple index out of range"))?,
            )?,
            _ => {
                return Err(gap(
                    "guide latent_shape requires original sequence admission",
                ));
            }
        };
        let mut count = 0;
        for local in 0..length {
            let pos = start + i128::from(local);
            if let Some(index) = window.iter().position(|v| i128::from(*v) == pos) {
                out.suffix_indices.push(
                    i64::try_from(suffix_base + i128::from(local))
                        .map_err(|_| gap("guide suffix exceeds native i64"))?,
                );
                out.keyframe_local_positions.push(index as i64);
                count += 1;
            }
        }
        if count > 0 {
            out.overlap_info.push((entry_idx as i64, count));
        }
        suffix_base += i128::from(length);
        let count = required(entry, "pre_filter_count")?.snapshot()?;
        match count {
            V::Float(_) => floating_token_offset = true,
            V::Int(_) | V::UInt(_) | V::Bool(_) => token_offset += i128::from(integer(&count)?),
            V::Tensor(_) | V::Quantized(_) => {
                return Err(gap(
                    "tensor guide token offsets require their original scalar index binding",
                ));
            }
            _ => {
                return Err(error(
                    "TypeError",
                    format!(
                        "unsupported operand type(s) for +=: 'int' and '{}'",
                        crate::backend_triton_fp8::typename(&count)
                    ),
                ));
            }
        }
    }
    Ok(out)
}
