//! Original Kitchen INT8 Hadamard cache and eager rotation operation order.
use crate::native::Error;
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use tch::{Device, Kind, Tensor};

type HadamardCache = Mutex<HashMap<(i64, Device, Kind), Tensor>>;
static HADAMARD_CACHE: LazyLock<HadamardCache> = LazyLock::new(|| Mutex::new(HashMap::new()));
// The frozen eager convrot_w4a4 module has its own independent mutable cache.
static EAGER_W4_HADAMARD_CACHE: LazyLock<HadamardCache> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn exception(class: &'static str, message: String) -> Error {
    Error::ReferenceException { class, message }
}

/// `_build_hadamard` for the typed, canonical CPU/CUDA device boundary.
/// Returns shared cached storage, including mutations visible to later calls.
pub fn hadamard(size: i64, device: Device, kind: Kind) -> Result<Tensor, Error> {
    cached_hadamard(&HADAMARD_CACHE, size, device, kind)
}
/// Original eager ConvRot W4A4 cache, separate from tensor.int8_utils and CUDA.
/// Repeated calls share storage within this original module only.
pub fn eager_w4_hadamard(size: i64, device: Device, kind: Kind) -> Result<Tensor, Error> {
    cached_hadamard(&EAGER_W4_HADAMARD_CACHE, size, device, kind)
}
fn cached_hadamard(
    cache: &HadamardCache,
    size: i64,
    device: Device,
    kind: Kind,
) -> Result<Tensor, Error> {
    let mut cache = cache.lock().unwrap();
    let key = (size, device, kind);
    if let Some(value) = cache.get(&key) {
        return Ok(value.shallow_clone());
    }
    if size < 4 || (size & (size - 1)) != 0 || (size as f64).log(4.).fract() != 0. {
        return Err(exception(
            "ValueError",
            format!("Regular Hadamard size must be a power of 4, got {size}"),
        ));
    }
    let h4 = Tensor::f_from_slice(&[1_i64, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1])?
        .f_reshape([4, 4])?
        .f_to_device_(device, kind, false, false)?;
    let mut h = h4.shallow_clone();
    let mut current_size = 4;
    while current_size < size {
        h = h.f_kron(&h4)?;
        current_size *= 4;
    }
    let normalized = h.f_div_scalar((size as f64).sqrt())?;
    cache.insert(key, normalized.shallow_clone());
    Ok(normalized)
}

fn groups(features: i64, group: i64, label: &str) -> Result<i64, Error> {
    if group == 0 {
        return Err(exception(
            "ZeroDivisionError",
            "integer modulo by zero".into(),
        ));
    }
    if features % group != 0 {
        return Err(exception(
            "ValueError",
            format!("{label} {features} not divisible by group_size {group}"),
        ));
    }
    Ok(features / group)
}

pub fn rotate_weight(weight: &Tensor, h: &Tensor, group: i64) -> Result<Tensor, Error> {
    let shape = weight.size();
    if shape.len() != 2 {
        let message = if shape.len() > 2 {
            "too many values to unpack (expected 2)".into()
        } else {
            format!(
                "not enough values to unpack (expected 2, got {})",
                shape.len()
            )
        };
        return Err(exception("ValueError", message));
    }
    let n_groups = groups(shape[1], group, "in_features")?;
    let grouped = weight.f_reshape([shape[0], n_groups, group])?;
    let h_t = h
        .f_transpose(0, 1)?
        .f_to_device_(weight.device(), weight.kind(), false, false)?;
    Ok(grouped.f_matmul(&h_t)?.f_reshape(shape.as_slice())?)
}

pub fn rotate_activation(x: &Tensor, h: &Tensor, group: i64) -> Result<Tensor, Error> {
    let shape = x.size();
    let features = shape
        .last()
        .ok_or_else(|| exception("IndexError", "tuple index out of range".into()))?;
    let n_groups = groups(*features, group, "features")?;
    let grouped = x.f_reshape([-1, n_groups, group])?;
    let h = h.f_to_device_(x.device(), x.kind(), false, false)?;
    Ok(grouped.f_matmul(&h)?.f_reshape(shape.as_slice())?)
}
