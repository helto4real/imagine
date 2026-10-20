//! Frozen comfy.lora and weight_adapter inference behavior. No Python runtime.
mod adapter_scalar;
mod adapters;
pub mod attributes;
pub mod bypass;
pub mod bypass_state;
pub mod deepcopy;
pub mod ejection;
pub mod injections;
mod loader;
pub mod model_access;
pub mod model_clone;
pub mod model_deepclone;
pub mod model_lifecycle;
pub mod model_patch_calls;
pub mod model_patcher;
pub mod model_pinning;
pub mod model_registry;
pub(crate) mod options_iterator;
mod owner;
mod pad_values;
pub use pad_values::pad_to_value_shape;
mod shape_metadata;
pub use shape_metadata::{calculate_shape_metadata, shape_metadata_matches, size_metadata};
pub mod pinned_key_set;
pub mod relationships;
mod tensor_scalar;
use crate::{
    execution_value::Value,
    native::{Error, Kitchen},
};
pub use adapters::Adapter;
pub use loader::{Loaded, Target, load_lora, qwen_key_map, text_key_map};
pub use owner::{PatchOwner, patch_weight};
use std::{collections::BTreeMap, sync::Arc};
use tch::{Kind, Tensor};
pub type Transform = Arc<dyn Fn(&Tensor) -> Result<Tensor, Error>>;
#[derive(Clone)]
pub enum PatchValue {
    Adapter(Adapter),
    Diff {
        weight: Value,
        pad: bool,
    },
    Set(Value),
    ModelAsLora(Value),
    Nested {
        base: Value,
        convert: Transform,
        patches: Vec<Patch>,
    },
    Unknown(String),
    Config(Value),
}
#[derive(Clone)]
pub struct Patch {
    pub strength: f64,
    pub value: PatchValue,
    pub model_strength: f64,
    pub offset: Option<[i64; 3]>,
    pub function: Option<Transform>,
}
impl Patch {
    pub fn new(value: PatchValue, strength: f64) -> Self {
        Self {
            strength,
            value,
            model_strength: 1.,
            offset: None,
            function: None,
        }
    }
}
pub type PatchMap = BTreeMap<String, Vec<Patch>>;
#[derive(Default)]
pub struct Diagnostics {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}
pub(crate) fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(crate) fn gap(message: impl Into<String>) -> Error {
    Error::BindingGap {
        backend: "weight_patching".into(),
        operation: message.into(),
    }
}
// Frozen Linux/CUDA profile: force_non_blocking=False, deterministic=False,
// no MPS/XPU/DirectML. The source helper therefore selects non_blocking=True.
pub fn cast_to_device(
    native: &Kitchen,
    t: &Tensor,
    device: tch::Device,
    kind: Kind,
    copy: bool,
) -> Result<Tensor, Error> {
    if t.device() == device {
        return Ok(t.f_to_device_(device, kind, false, copy)?);
    }
    // Match cast_to's empty_like + copy_, including Preserve memory format.
    let mut out = unsafe {
        crate::boxed_aten::call(
            native,
            "aten::empty_like",
            "",
            &[Value::tensor(t.shallow_clone())],
            &vec![
                ("device".into(), Value::Device(device)),
                ("dtype".into(), Value::DType(kind)),
            ],
        )?
    }
    .as_tensor()?;
    native.copy_tensor(&mut out, t, true)?;
    Ok(out)
}
pub(crate) fn cast(
    native: &Kitchen,
    t: &Tensor,
    like: &Tensor,
    kind: Kind,
) -> Result<Tensor, Error> {
    cast_to_device(native, t, like.device(), kind, false)
}
pub fn string_to_seed(s: &str) -> u64 {
    let mut crc = 0xffff_ffff_u64;
    for c in s.chars() {
        crc ^= c as u64;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    crc ^ 0xffff_ffff
}
pub fn pad_to_shape(t: &Tensor, shape: &[i64]) -> Result<Tensor, Error> {
    let mut smaller = false;
    for (i, n) in shape.iter().enumerate() {
        let old = t
            .size()
            .get(i)
            .copied()
            .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
        smaller |= *n < old;
    }
    // any([ ... ]) first builds the entire source list. A later bad index
    // therefore takes precedence over an earlier shrinking dimension.
    if smaller {
        return Err(exception(
            "ValueError",
            "The new shape must be larger than the original tensor in all dimensions",
        ));
    }
    if shape.len() != t.dim() {
        return Err(exception(
            "ValueError",
            "The new shape must have the same number of dimensions as the original tensor",
        ));
    }
    let output = Tensor::f_zeros(shape, (t.kind(), t.device()))?;
    let mut view = output.shallow_clone();
    for (i, n) in t.size().iter().enumerate() {
        view = view.f_narrow(i as i64, 0, *n)?;
    }
    view.f_copy_(t)?;
    Ok(output)
}
fn transform(p: &Patch, t: &Tensor) -> Result<Tensor, Error> {
    p.function
        .as_ref()
        .map_or_else(|| Ok(t.shallow_clone()), |f| f(t))
}
/// The original mutates its supplied weight/views; callers must provide a copy
/// when patching must not modify shared original storage.
pub fn calculate_weight(
    native: &Kitchen,
    patches: &[Patch],
    mut weight: Tensor,
    key: &str,
    intermediate: Kind,
    originals: Option<&BTreeMap<String, Value>>,
    diagnostics: &mut Diagnostics,
) -> Result<Tensor, Error> {
    for p in patches {
        let old = if let Some([dim, start, len]) = p.offset {
            let old = weight.shallow_clone();
            weight = weight.f_narrow(dim, start, len)?;
            Some(old)
        } else {
            None
        };
        if p.model_strength != 1. {
            let _ = weight.f_mul_scalar_(p.model_strength)?;
        }
        let nested;
        let value = if let PatchValue::Nested {
            base,
            convert,
            patches,
        } = &p.value
        {
            let base = cast_to_device(
                native,
                &base.as_tensor()?,
                weight.device(),
                intermediate,
                true,
            )?;
            nested = PatchValue::Diff {
                weight: Value::tensor(calculate_weight(
                    native,
                    patches,
                    convert(&base)?,
                    key,
                    intermediate,
                    None,
                    diagnostics,
                )?),
                pad: false,
            };
            &nested
        } else {
            &p.value
        };
        match value {
            PatchValue::Adapter(a) => {
                weight = a.calculate(native, weight, p, key, intermediate, diagnostics)?;
            }
            PatchValue::Diff { weight: diff, pad } => {
                let diff = diff.as_tensor()?;
                if *pad && diff.size() != weight.size() {
                    weight = pad_to_shape(&weight, &diff.size())?;
                }
                if p.strength != 0. {
                    if diff.size() != weight.size() {
                        diagnostics
                            .warnings
                            .push(format!("WARNING SHAPE MISMATCH {key} WEIGHT NOT MERGED torch.Size({:?}) != torch.Size({:?})",diff.size(),weight.size()));
                    } else {
                        let delta = cast(native, &diff, &weight, weight.kind())?
                            .f_mul_scalar(p.strength)?;
                        let _ = weight.f_add_(&transform(p, &delta)?)?;
                    }
                }
            }
            PatchValue::Set(v) => weight.f_copy_(&v.as_tensor()?)?,
            PatchValue::ModelAsLora(target) => {
                let orig = originals
                    .and_then(|m| m.get(key))
                    .ok_or_else(|| exception("KeyError", format!("'{key}'")))?
                    .as_tensor()?;
                let diff = cast(native, &target.as_tensor()?, &weight, intermediate)?
                    .f_sub(&cast(native, &orig, &weight, intermediate)?)?;
                let delta =
                    cast(native, &diff, &weight, weight.kind())?.f_mul_scalar(p.strength)?;
                let _ = weight.f_add_(&transform(p, &delta)?)?;
            }
            PatchValue::Unknown(name) => diagnostics
                .warnings
                .push(format!("patch type not recognized {name} {key}")),
            PatchValue::Config(_) => diagnostics
                .warnings
                .push(format!("patch type not recognized config {key}")),
            PatchValue::Nested { .. } => unreachable!(),
        }
        if let Some(old) = old {
            weight = old;
        }
    }
    Ok(weight)
}
pub fn calculate_shape(patches: &[Patch], weight: &Tensor) -> Result<Vec<i64>, Error> {
    calculate_shape_from_shape(patches, weight.size())
}
pub fn calculate_shape_from_shape(patches: &[Patch], shape: Vec<i64>) -> Result<Vec<i64>, Error> {
    shape_metadata::integer_dimensions(&calculate_shape_metadata(patches, size_metadata(&shape))?)
}
pub mod extensions;
pub mod options;

pub mod hook_patches;
pub mod hooks;
