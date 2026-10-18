//! Original CUDA ConvRot wrapper allocations and launch operands.
//! These bindings do not select a quantizer or replace high-level heuristics.
use crate::native::{Error, KernelArg, Kitchen};
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use tch::{Device, Kind, Tensor};

type EmptyCache = Mutex<HashMap<(Device, Kind), Tensor>>;
static EMPTY_CACHE: LazyLock<EmptyCache> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn stream(native: &Kitchen, x: &Tensor) -> Result<i64, Error> {
    match x.device() {
        Device::Cuda(index) => Ok(native.current_stream(index)? as i64),
        _ => Err(Error::Native("CUDA ConvRot requires CUDA input".into())),
    }
}

fn empty(device: Device, kind: Kind) -> Result<Tensor, Error> {
    let mut cache = EMPTY_CACHE.lock().unwrap();
    if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry((device, kind)) {
        entry.insert(Tensor::f_empty([0], (kind, device))?);
    }
    Ok(cache[&(device, kind)].shallow_clone())
}

/// Typed internal operands for Kitchen's four fused activation codes.
#[derive(Clone, Copy)]
pub enum InputActivation<'a> {
    None,
    GeluTanh,
    SwiGlu,
    RmsNormDefault {
        weight: Option<&'a Tensor>,
    },
    RmsNorm {
        weight: Option<&'a Tensor>,
        epsilon: f64,
    },
}

/// # Safety
/// x must be contiguous 2D CUDA storage on the current stream, satisfying the
/// original kernel's scalar ranges and allocation limits. This wrapper does
/// not add validation or a contiguous copy absent from the original wrapper.
pub unsafe fn rotate_weight(native: &Kitchen, x: &Tensor, group: i64) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    let output = x.f_empty_like()?;
    let stream = stream(native, x)?;
    unsafe {
        native.call_unchecked(
            "rotate_int8_convrot_weight",
            &[T(x), T(&output), Int(group), Int(stream)],
        )?;
    }
    Ok(output)
}

/// # Safety
/// Same storage/stream/scalar requirements as rotate_weight. Original native
/// validation and seed>0 stochastic policy are retained without reshaping x.
pub unsafe fn rowwise(
    native: &Kitchen,
    x: &Tensor,
    group: i64,
    seed: impl Into<crate::rng_seed::NativeSeed>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    unsafe { rowwise_with_seed(native, x, group, || Ok(seed)) }
}

/// # Safety
/// Retains the original allocation and launch contract with deferred seed parsing.
pub(crate) unsafe fn rowwise_with_seed(
    native: &Kitchen,
    x: &Tensor,
    group: i64,
    seed: impl FnOnce() -> Result<crate::rng_seed::NativeSeed, Error>,
) -> Result<(Tensor, Tensor), Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let output = native.empty_like_dtype(x, Kind::Int8)?;
    let scales = Tensor::f_empty([x.size()[0], 1], (Kind::Float, x.device()))?;
    let stream = stream(native, x)?;
    let seed = seed()?;
    unsafe {
        native.call_unchecked(
            "quantize_int8_rowwise_convrot",
            &[
                T(x),
                T(&output),
                T(&scales),
                Int(group),
                Bool(seed.enabled),
                Int(seed.bits as i64),
                Int(stream),
            ],
        )?;
    }
    Ok((output, scales))
}

/// # Safety
/// Same requirements as rowwise; group must also be positive for the original
/// partial-absmax allocation. Rotated and partial buffers stay live on stream.
pub unsafe fn staged(
    native: &Kitchen,
    x: &Tensor,
    group: i64,
    seed: impl Into<crate::rng_seed::NativeSeed>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    unsafe { staged_with_seed(native, x, group, || Ok(seed)) }
}

/// # Safety
/// Retains the original allocation and launch contract with deferred seed parsing.
pub(crate) unsafe fn staged_with_seed(
    native: &Kitchen,
    x: &Tensor,
    group: i64,
    seed: impl FnOnce() -> Result<crate::rng_seed::NativeSeed, Error>,
) -> Result<(Tensor, Tensor), Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let groups = x.size()[x.dim() - 1] / group;
    let rotated = x.f_empty_like()?;
    let partial = Tensor::f_empty([x.size()[0], groups], (Kind::Float, x.device()))?;
    let output = native.empty_like_dtype(x, Kind::Int8)?;
    let scales = Tensor::f_empty([x.size()[0], 1], (Kind::Float, x.device()))?;
    let stream = stream(native, x)?;
    let seed = seed()?;
    unsafe {
        native.call_unchecked(
            "quantize_int8_convrot_staged",
            &[
                T(x),
                T(&rotated),
                T(&partial),
                T(&output),
                T(&scales),
                Int(group),
                Bool(seed.enabled),
                Int(seed.bits as i64),
                Int(stream),
            ],
        )?;
    }
    Ok((output, scales))
}

/// # Safety
/// x must satisfy rowwise's storage/stream contract; input width must include
/// both gate/up halves for SwiGLU. RMS weights must be valid ATen tensors ready
/// for the same device's current stream. Original conversion/validation applies.
pub unsafe fn rowwise64(
    native: &Kitchen,
    x: &Tensor,
    group: i64,
    seed: impl Into<crate::rng_seed::NativeSeed>,
    activation: InputActivation<'_>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    unsafe { rowwise64_with_seed(native, x, group, || Ok(seed), activation) }
}

/// # Safety
/// Retains the original allocation and launch contract with deferred seed parsing.
pub(crate) unsafe fn rowwise64_with_seed(
    native: &Kitchen,
    x: &Tensor,
    group: i64,
    seed: impl FnOnce() -> Result<crate::rng_seed::NativeSeed, Error>,
    activation: InputActivation<'_>,
) -> Result<(Tensor, Tensor), Error> {
    use KernelArg::{Bool, Float, Int, Tensor as T};
    let width_factor = if matches!(activation, InputActivation::SwiGlu) {
        2
    } else {
        1
    };
    let output = Tensor::f_empty(
        [x.size()[0], x.size()[1] / width_factor],
        (Kind::Int8, x.device()),
    )?;
    let scales = Tensor::f_empty([x.size()[0], 1], (Kind::Float, x.device()))?;
    let stream = stream(native, x)?;
    let weight = match &activation {
        InputActivation::RmsNorm {
            weight: Some(weight),
            ..
        }
        | InputActivation::RmsNormDefault {
            weight: Some(weight),
        } => weight
            .f_to_device_(x.device(), x.kind(), false, false)?
            .f_contiguous()?,
        _ => empty(x.device(), x.kind())?,
    };
    let (code, epsilon) = match activation {
        InputActivation::None => (0, 0.),
        InputActivation::GeluTanh => (1, 0.),
        InputActivation::SwiGlu => (2, 0.),
        InputActivation::RmsNorm { epsilon, .. } => (3, epsilon),
        InputActivation::RmsNormDefault { .. } => {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "float() argument must be a string or a real number, not 'NoneType'"
                    .into(),
            });
        }
    };
    let seed = seed()?;
    unsafe {
        native.call_unchecked(
            "quantize_int8_rowwise_convrot64",
            &[
                T(x),
                T(&output),
                T(&scales),
                Int(group),
                Bool(seed.enabled),
                Int(code),
                Int(seed.bits as i64),
                T(&weight),
                Float(epsilon),
                Int(stream),
            ],
        )?;
    }
    Ok((output, scales))
}

/// CUDA backend dequantize_int8_simple_dtype, including its original eager
/// broadcasting fallback and scalar/elementwise/rowwise mode precedence.
pub fn dequantize_simple(
    native: &Kitchen,
    q: &Tensor,
    scale: &Tensor,
    kind: Kind,
) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    let q = q.f_contiguous()?;
    let shape = q.size();
    let scale_shape = scale.size();
    let mode = if scale.numel() == 1 {
        0
    } else if scale_shape == shape {
        1
    } else if !shape.is_empty()
        && scale.dim() == q.dim()
        && scale_shape[..scale.dim() - 1] == shape[..q.dim() - 1]
        && scale_shape[scale.dim() - 1] == 1
    {
        2
    } else {
        -1
    };
    if mode < 0 {
        return Ok(q.f_to_kind(Kind::Float)?.f_mul(scale)?.f_to_kind(kind)?);
    }
    let scale = scale
        .f_to_device_(q.device(), Kind::Float, false, false)?
        .f_contiguous()?;
    let output = Tensor::f_empty(shape.as_slice(), (kind, q.device()))?;
    let stream = stream(native, &q)?;
    unsafe {
        native.call_unchecked(
            "dequantize_int8_simple",
            &[
                T(&q),
                T(&scale.f_reshape([-1])?),
                T(&output),
                Int(*shape.last().unwrap_or(&1)),
                Int(mode),
                Int(stream),
            ],
        )?;
    }
    Ok(output)
}

static MAX_SHARED_CACHE: LazyLock<Mutex<HashMap<usize, i64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const FUSED_MAX_K: i64 = 16384;

/// Original `_max_dynamic_shared_memory_per_block`, keyed by CUDA device.
pub fn max_dynamic_shared_memory(native: &Kitchen, device: usize) -> Result<i64, Error> {
    let mut cache = MAX_SHARED_CACHE.lock().unwrap();
    if let Some(limit) = cache.get(&device) {
        return Ok(*limit);
    }
    let props = native.device_properties(device)?;
    let limit = if (props.major, props.minor) == (7, 5) {
        props.shared_memory_per_block
    } else {
        props.shared_memory_per_block_optin
    };
    cache.insert(device, limit);
    Ok(limit)
}

pub fn fused_shared_memory_bytes(rows: i64, width: i64) -> i64 {
    let block_threads = if rows == 1 {
        512
    } else {
        match width {
            256 => 64,
            2560 => 640,
            6144 => 768,
            _ => 1024,
        }
    };
    (width + (block_threads / 64) * 2 * 256) * 4
}

fn shared_memory_fits(native: &Kitchen, x: &Tensor, width: i64, group: i64) -> Result<bool, Error> {
    match x.device() {
        Device::Cuda(device) if group == 256 => Ok(fused_shared_memory_bytes(x.size()[0], width)
            < max_dynamic_shared_memory(native, device)?),
        _ => Ok(true),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPath {
    Rowwise64,
    Staged,
    Rowwise,
    EagerRotation,
}

/// Original high-level CUDA weight selection, including its strict `<` shared
/// memory check and original branch order. No autotuning is introduced here.
pub fn weight_path(native: &Kitchen, x: &Tensor, group: i64) -> Result<WeightPath, Error> {
    let width = *x.size().last().ok_or_else(|| Error::ReferenceException {
        class: "IndexError",
        message: "tuple index out of range".into(),
    })?;
    if group == 256
        && width % 256 == 0
        && (256..=FUSED_MAX_K).contains(&width)
        && shared_memory_fits(native, x, width, group)?
    {
        return Ok(WeightPath::Rowwise64);
    }
    if group == 256 && width % 256 == 0 && (1024..8192).contains(&width) {
        return Ok(WeightPath::Staged);
    }
    if group == 256
        && width % 256 == 0
        && width <= FUSED_MAX_K
        && (width <= 5120 || width >= 8192)
        && shared_memory_fits(native, x, width, group)?
    {
        return Ok(WeightPath::Rowwise);
    }
    if group == 256 && width % 256 == 0 {
        return Ok(WeightPath::Staged);
    }
    Ok(WeightPath::EagerRotation)
}

/// CUDA backend's high-level `quantize_int8_convrot_weight`.
/// # Safety
/// CUDA storage must satisfy the selected original wrapper's launch contract.
/// The ordinary registry supplies its dtype/rank/device validation separately.
pub unsafe fn quantize_weight(
    native: &Kitchen,
    weight: &Tensor,
    group: i64,
    seed: impl Into<crate::rng_seed::NativeSeed>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    unsafe { quantize_weight_with_seed(native, weight, group, |_| Ok(seed)) }
}
/// # Safety
/// The original selected launch contract applies, including delayed scalar parsing.
pub(crate) unsafe fn quantize_weight_with_seed(
    native: &Kitchen,
    weight: &Tensor,
    group: i64,
    seed: impl FnOnce(&str) -> Result<crate::rng_seed::NativeSeed, Error>,
) -> Result<(Tensor, Tensor), Error> {
    if weight.dim() != 2 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "ConvRot INT8 weight quantization expects a 2D weight tensor".into(),
        });
    }
    let weight = weight.f_contiguous()?;
    match weight_path(native, &weight, group)? {
        WeightPath::Rowwise64 => unsafe {
            rowwise64_with_seed(
                native,
                &weight,
                group,
                || seed("quantize_int8_rowwise_convrot64"),
                InputActivation::None,
            )
        },
        WeightPath::Staged => unsafe {
            staged_with_seed(native, &weight, group, || {
                seed("quantize_int8_convrot_staged")
            })
        },
        WeightPath::Rowwise => unsafe {
            rowwise_with_seed(native, &weight, group, || {
                seed("quantize_int8_rowwise_convrot")
            })
        },
        WeightPath::EagerRotation => {
            let h = crate::int8_utils::hadamard(group, weight.device(), weight.kind())?;
            let rotated = crate::int8_utils::rotate_weight(&weight, &h, group)?;
            crate::quantization::int8_rowwise_with_seed(native, &rotated, || {
                seed("quantize_int8_rowwise").map(|s| (s.enabled, s.bits as i64))
            })
        }
    }
}

/// CUDA backend's `dequantize_int8_convrot_weight_dtype`. Nonfused rotation
/// retains the original FP32 dequantization/matmul before the final dtype cast.
/// # Safety
/// q must satisfy the selected original native payload/scale/stream contract.
pub unsafe fn dequantize_weight(
    native: &Kitchen,
    q: &Tensor,
    scale: &Tensor,
    group: i64,
    kind: Kind,
) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    if q.dim() != 2 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "ConvRot INT8 weight dequantization expects a 2D q tensor".into(),
        });
    }
    let q = q.f_contiguous()?;
    let width = q.size()[1];
    if [64, 256].contains(&group) && width % group == 0 && width <= FUSED_MAX_K {
        let scale = scale
            .f_to_device_(q.device(), Kind::Float, false, false)?
            .f_reshape([-1])?
            .f_contiguous()?;
        let output = Tensor::f_empty(q.size(), (kind, q.device()))?;
        let stream = stream(native, &q)?;
        unsafe {
            native.call_unchecked(
                "dequantize_int8_convrot_weight",
                &[T(&q), T(&scale), T(&output), Int(group), Int(stream)],
            )?;
        }
        return Ok(output);
    }
    let h = crate::int8_utils::hadamard(group, q.device(), Kind::Float)?;
    let decoded = dequantize_simple(native, &q, scale, Kind::Float)?;
    Ok(crate::int8_utils::rotate_weight(&decoded, &h, group)?.f_to_kind(kind)?)
}

/// Original CUDA quantize_and_rotate_rowwise fallback: eager activation
/// rotation, then the unchanged CUDA rowwise INT8 quantizer.
pub fn quantize_activation(
    native: &Kitchen,
    x: &Tensor,
    h: &Tensor,
    group: i64,
    seed: impl Into<crate::rng_seed::NativeSeed>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    let rotated = crate::int8_utils::rotate_activation(x, h, group)?;
    crate::quantization::int8_rowwise(native, &rotated, seed)
}
