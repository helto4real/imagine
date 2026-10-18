//! Frozen CUDA linear wrappers; individual routes do not advertise a full backend.
use crate::{
    convrot::InputActivation,
    int8_linear,
    native::{Error, KernelArg, Kitchen},
};
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use tch::{Device, Kind, Tensor};

static EMPTY: LazyLock<Mutex<HashMap<(Device, Kind), Tensor>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn empty(device: Device, kind: Kind) -> Result<Tensor, Error> {
    let mut cache = EMPTY.lock().unwrap();
    if let Some(tensor) = cache.get(&(device, kind)) {
        return Ok(tensor.shallow_clone());
    }
    let tensor = Tensor::f_empty([0], (kind, device))?;
    cache.insert((device, kind), tensor.shallow_clone());
    Ok(tensor)
}

fn aligned16(tensor: &Tensor) -> bool {
    (tensor.data_ptr() as usize).is_multiple_of(16)
}

pub(crate) fn vector(
    native: &Kitchen,
    tensor: &Tensor,
    device: Device,
    kind: Kind,
) -> Result<Tensor, Error> {
    if tensor.device() == device
        && tensor.kind() == kind
        && tensor.is_contiguous()
        && aligned16(tensor)
    {
        return Ok(tensor.shallow_clone());
    }
    let value = tensor
        .f_to_device_(device, kind, false, false)?
        .f_contiguous()?;
    if aligned16(&value) {
        Ok(value)
    } else {
        native.clone_tensor(&value)
    }
}

/// # Safety
/// Selected native operands must satisfy the original CUDA wrapper contract;
/// this typed entry point retains the original checks and fallback decisions.
pub unsafe fn fp16_linear(
    native: &Kitchen,
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    residual: Option<&Tensor>,
    residual_scale: Option<&Tensor>,
) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    if residual.is_some() && residual_scale.is_none() {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "fp16_linear: residual requires residual_scale".into(),
        });
    }
    let shape = x.size();
    let x_2d = if x.dim() == 2 && x.is_contiguous() {
        x.shallow_clone()
    } else {
        let width = shape.last().ok_or_else(|| Error::ReferenceException {
            class: "IndexError",
            message: "tuple index out of range".into(),
        })?;
        x.f_reshape([-1, *width])?.f_contiguous()?
    };
    let m = x_2d.size()[0];
    let n = *weight
        .size()
        .first()
        .ok_or_else(|| Error::ReferenceException {
            class: "IndexError",
            message: "tuple index out of range".into(),
        })?;
    let supported = x.kind() == Kind::Half
        && weight.kind() == Kind::Half
        && matches!(weight.device(), Device::Cuda(_))
        && weight.device() == x.device()
        && aligned16(&x_2d)
        && aligned16(weight)
        && bias.is_none_or(|b| b.kind() == Kind::Half)
        && residual.is_none_or(|r| {
            r.kind() == Kind::Half && residual_scale.unwrap().kind() == Kind::Half && aligned16(r)
        });
    if !supported {
        return int8_linear::fp16_linear(x, weight, bias, residual, residual_scale);
    }
    let weight = weight.f_contiguous()?;
    let Device::Cuda(device) = x.device() else {
        unreachable!("CUDA support gate checked")
    };
    let stream = native.current_stream(device)? as i64;
    let mut out = Tensor::f_empty([m, n], (Kind::Half, x.device()))?;
    let bias_arg = match bias {
        Some(bias) => vector(native, bias, x.device(), Kind::Half)?,
        None => empty(x.device(), Kind::Half)?,
    };
    let residual_2d = residual
        .map(|r| r.f_reshape([m, n])?.f_contiguous())
        .transpose()?;
    let fused = if let Some(residual) = &residual_2d {
        let scale = vector(native, residual_scale.unwrap(), x.device(), Kind::Half)?;
        unsafe {
            native.call_unchecked(
                "cutlass_fp16_linear_residual",
                &[
                    T(&x_2d),
                    T(&weight),
                    T(&bias_arg),
                    T(&scale),
                    T(residual),
                    T(&out),
                    Int(stream),
                ],
            )?
        }
        .value
            != 0
    } else {
        false
    };
    if !fused {
        let ok = unsafe {
            native.call_unchecked(
                "cutlass_fp16_linear",
                &[T(&x_2d), T(&weight), T(&bias_arg), T(&out), Int(stream)],
            )?
        }
        .value
            != 0;
        if !ok {
            out = x_2d.f_linear(&weight, bias)?;
        }
        out = int8_linear::apply_residual(&out, residual_2d.as_ref(), residual_scale)?;
    }
    if shape.len() == 2 {
        Ok(out)
    } else {
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(n);
        Ok(out.f_reshape(output_shape.as_slice())?)
    }
}

pub struct Int8M1<'a> {
    pub x: &'a Tensor,
    pub weight: &'a Tensor,
    pub weight_scale: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub output_kind: Kind,
    pub convrot: bool,
    pub convrot_group: i64,
    pub residual: Option<&'a Tensor>,
    pub residual_scale: Option<&'a Tensor>,
}

fn gemv_residual_args(
    native: &Kitchen,
    request: &Int8M1<'_>,
    n: i64,
) -> Result<Option<(Tensor, Tensor)>, Error> {
    let x = request.x;
    Ok(match (request.residual, request.residual_scale) {
        (Some(residual), Some(scale))
            if residual.kind() == request.output_kind
                && scale.kind() == request.output_kind
                && residual.numel() as i64 == n
                && scale.numel() as i64 == n =>
        {
            Some((
                residual.f_reshape([1, n])?.f_contiguous()?,
                vector(native, scale, x.device(), request.output_kind)?,
            ))
        }
        _ => None,
    })
}

/// The same no-activation operands for the later GEMV route.
pub type Int8Gemv<'a> = Int8M1<'a>;

/// Original later GEMV predicate, excluding the earlier no-activation M=1 route.
pub fn int8_gemv_supported(
    native: &Kitchen,
    x_2d: &Tensor,
    weight: &Tensor,
    convrot: bool,
    group: i64,
) -> Result<bool, Error> {
    let (m, k) = (x_2d.size()[0], x_2d.size()[1]);
    Ok(!int8_m1_supported(native, x_2d, weight, convrot, group)?
        && ((m == 1 && k % 4 == 0) || (m == 2 && k % 16 == 0)))
}

/// Original later `int8_linear` GEMV branch, with input_act None.
/// # Safety
/// Caller must select this original route using int8_gemv_supported, with
/// floating CUDA input/output, 2D INT8 weight and original valid launch operands.
/// This does not replace activation or GEMM routes. Unwritten empty outputs
/// may only be inspected under the separately approved controlled-state bounds.
pub unsafe fn int8_gemv_linear(native: &Kitchen, request: &Int8Gemv<'_>) -> Result<Tensor, Error> {
    unsafe {
        int8_dispatch(
            native,
            request,
            LinearRoute::Gemv,
            InputActivation::None,
            None,
        )
    }
}

/// Original remaining no-activation predicate after M=1 and GEMV precedence.
pub fn int8_gemm_supported(
    native: &Kitchen,
    x: &Tensor,
    weight: &Tensor,
    convrot: bool,
    group: i64,
) -> Result<bool, Error> {
    Ok(!int8_m1_supported(native, x, weight, convrot, group)?
        && !int8_gemv_supported(native, x, weight, convrot, group)?)
}

/// Original non-Turing CUDA GEMM route on the initial SM89 platform.
/// `disable_cutlass` models the frozen module's COMFY_KITCHEN_DISABLE_CUTLASS
/// value; it does not replace the original default or introduce tuning policy.
/// # Safety
/// Select int8_gemm_supported first. Same floating/INT8 operands and stream
/// contract as GEMV; current CUDA device must be the input device on SM89.
/// Turing kernels/padding and complete backend initialization remain pending.
/// Zero-K output inspection is limited to the separately approved 12 cases.
pub unsafe fn int8_gemm_linear(
    native: &Kitchen,
    request: &Int8M1<'_>,
    disable_cutlass: bool,
) -> Result<Tensor, Error> {
    unsafe {
        int8_dispatch(
            native,
            request,
            LinearRoute::Gemm { disable_cutlass },
            InputActivation::None,
            None,
        )
    }
}

type BeforeDequant<'a> = dyn FnMut(&Tensor, &Tensor) -> Result<(), Error> + 'a;

/// Development-only metadata observer at the original cuBLAS consumer boundary.
/// No observer is installed by the ordinary selected-route adapter.
/// # Safety
/// Same SM89/current-device/native operand requirements as int8_gemm_linear.
/// The observer may inspect metadata only and must refuse unmatched storage
/// before returning success; it must not read/fill either tensor or alter stream
/// state. D-05-INT8-CUDA-CUBLAS-EMPTY-INTERMEDIATES bounds zero-K validation.
#[doc(hidden)]
pub unsafe fn int8_gemm_linear_observed(
    native: &Kitchen,
    request: &Int8M1<'_>,
    disable_cutlass: bool,
    observer: &mut BeforeDequant<'_>,
) -> Result<Tensor, Error> {
    let mut bridge = |consumer: Int8Consumer, tensors: &[&Tensor]| {
        if consumer == Int8Consumer::Dequantize {
            observer(tensors[0], tensors[1])
        } else {
            Ok(())
        }
    };
    unsafe {
        int8_dispatch(
            native,
            request,
            LinearRoute::Gemm { disable_cutlass },
            InputActivation::None,
            Some(&mut bridge),
        )
    }
}

/// The two original consumer boundaries used only by development guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int8Consumer {
    Dequantize,
    UnwrittenOutput,
    RmsTemporary { start: usize, bytes: usize },
}
pub type Int8Observer<'a> = dyn FnMut(Int8Consumer, &[&Tensor]) -> Result<(), Error> + 'a;

/// Original typed SM89 CUDA INT8 activation and M=1/GEMV/GEMM selection.
/// # Safety
/// CUDA floating input/output and 2D INT8 weight must meet the original native
/// operand contract, on the current SM89 device. Generic kwargs and Turing
/// policy/initialization are separate remaining Stage 1 work.
pub unsafe fn int8_linear(
    native: &Kitchen,
    request: &int8_linear::Linear<'_>,
    disable_cutlass: bool,
) -> Result<Tensor, Error> {
    unsafe { int8_linear_observed_impl(native, request, disable_cutlass, None) }
}

/// Development-only guards before dequantization and no-write output consumption.
/// # Safety
/// Same requirements as int8_linear. The observer must inspect metadata only,
/// refuse unmatched storage, and never read/fill tensors or alter stream state.
/// D-05-INT8-CUDA-LINEAR-ZERO-K-FAMILY governs controlled zero-K acceptance.
#[doc(hidden)]
pub unsafe fn int8_linear_observed(
    native: &Kitchen,
    request: &int8_linear::Linear<'_>,
    disable_cutlass: bool,
    observer: &mut Int8Observer<'_>,
) -> Result<Tensor, Error> {
    unsafe { int8_linear_observed_impl(native, request, disable_cutlass, Some(observer)) }
}

unsafe fn int8_linear_observed_impl(
    native: &Kitchen,
    request: &int8_linear::Linear<'_>,
    disable_cutlass: bool,
    observer: Option<&mut Int8Observer<'_>>,
) -> Result<Tensor, Error> {
    let operands = Int8M1 {
        x: request.x,
        weight: request.weight,
        weight_scale: request.weight_scale,
        bias: request.bias,
        output_kind: request.output_kind,
        convrot: request.convrot,
        convrot_group: request.convrot_group,
        residual: request.residual,
        residual_scale: request.residual_scale,
    };
    unsafe {
        int8_dispatch(
            native,
            &operands,
            LinearRoute::Auto { disable_cutlass },
            request.input_activation,
            observer,
        )
    }
}

#[derive(Clone, Copy)]
enum LinearRoute {
    M1,
    Auto { disable_cutlass: bool },
    Gemv,
    Gemm { disable_cutlass: bool },
}

static CUBLAS_WORKSPACES: LazyLock<Mutex<HashMap<usize, Tensor>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CUTLASS_DEVICES: LazyLock<Mutex<HashMap<usize, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn supports_cutlass(native: &Kitchen, device: usize) -> Result<bool, Error> {
    let mut cache = CUTLASS_DEVICES.lock().unwrap();
    if let Some(supported) = cache.get(&device) {
        return Ok(*supported);
    }
    let supported = native.device_properties(device)?.major >= 8;
    cache.insert(device, supported);
    Ok(supported)
}

pub(crate) fn cublas_workspace(native: &Kitchen, device: usize) -> Result<Tensor, Error> {
    let mut cache = CUBLAS_WORKSPACES.lock().unwrap();
    if let Some(tensor) = cache.get(&device) {
        return Ok(tensor.shallow_clone());
    }
    let bytes = if native.device_properties(device)?.major >= 9 {
        33_554_432
    } else {
        4_194_304
    };
    let tensor = Tensor::f_empty([bytes], (Kind::Uint8, Device::Cuda(device)))?;
    cache.insert(device, tensor.shallow_clone());
    Ok(tensor)
}

fn fused_shared_fits(native: &Kitchen, device: usize, m: i64, k: i64) -> Result<bool, Error> {
    Ok(crate::convrot::fused_shared_memory_bytes(m, k)
        < crate::convrot::max_dynamic_shared_memory(native, device)?)
}
pub(crate) fn input_activation_fuses(
    native: &Kitchen,
    device: usize,
    m: i64,
    k: i64,
    group: i64,
    convrot: bool,
) -> Result<bool, Error> {
    Ok(convrot
        && group == 256
        && k % 256 == 0
        && (256..=16384).contains(&k)
        && fused_shared_fits(native, device, m, k)?)
}

unsafe fn int8_dispatch(
    native: &Kitchen,
    request: &Int8M1<'_>,
    route: LinearRoute,
    mut activation: InputActivation<'_>,
    mut observer: Option<&mut Int8Observer<'_>>,
) -> Result<Tensor, Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    if request.residual.is_some() && request.residual_scale.is_none() {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "int8_linear: residual requires residual_scale".into(),
        });
    }
    let x = request.x;
    let shape = x.size();
    let mut x_2d = if x.dim() == 2 && x.is_contiguous() {
        x.shallow_clone()
    } else {
        let width = shape.last().ok_or_else(|| Error::ReferenceException {
            class: "IndexError",
            message: "tuple index out of range".into(),
        })?;
        x.f_reshape([-1, *width])?.f_contiguous()?
    };
    let mut weight = request.weight.f_contiguous()?;
    let Device::Cuda(device) = x.device() else {
        return Err(Error::Native("CUDA GEMV requires CUDA input".into()));
    };
    let stream = native.current_stream(device)? as i64;
    let m = x_2d.size()[0];
    let k = x_2d.size()[1]
        / if matches!(activation, InputActivation::SwiGlu | InputActivation::Situ) {
            2
        } else {
            1
        };
    let (n, kw) = (weight.size()[0], weight.size()[1]);
    let group = request.convrot_group;
    let fused = input_activation_fuses(native, device, m, k, group, request.convrot)?;
    if !matches!(activation, InputActivation::None) && !fused {
        if matches!(
            activation,
            InputActivation::RmsNorm { .. } | InputActivation::RmsNormDefault { .. }
        ) && let Some(callback) = observer.as_mut()
        {
            let trace = native.rms_metadata_observer()?;
            let activated = int8_linear::apply_input(&x_2d, &activation);
            let temporary = trace.finish();
            x_2d = activated?;
            let temporary = temporary?;
            callback(
                Int8Consumer::RmsTemporary {
                    start: temporary.start as usize,
                    bytes: temporary.bytes as usize,
                },
                &[],
            )?;
        } else {
            x_2d = int8_linear::apply_input(&x_2d, &activation)?;
        }
        activation = InputActivation::None;
    }
    if k != kw {
        return Err(Error::ReferenceException {
            class: "AssertionError",
            message: "Input and weight inner dimensions must match".into(),
        });
    }
    let output_code = match request.output_kind {
        Kind::Float => 0,
        Kind::Half => 1,
        Kind::BFloat16 => 2,
        _ => {
            return Err(Error::Native(
                "GEMV output requires original floating dtype contract".into(),
            ));
        }
    };
    let route = match route {
        LinearRoute::Auto { disable_cutlass } => {
            if ((request.convrot && fused) || matches!(activation, InputActivation::None))
                && int8_m1_supported_width(native, &x_2d, &weight, request.convrot, group, k)?
            {
                LinearRoute::M1
            } else if (m == 1 && k % 4 == 0) || (m == 2 && k % 16 == 0) {
                LinearRoute::Gemv
            } else {
                LinearRoute::Gemm { disable_cutlass }
            }
        }
        other => other,
    };
    let mut fused_residual_done = false;
    let mut dequantized_output = false;
    let output = if matches!(route, LinearRoute::M1) {
        let quantized = if request.convrot {
            empty(x.device(), Kind::Int8)?
        } else {
            Tensor::f_empty([1, k], (Kind::Int8, x.device()))?
        };
        let x_scale = if request.convrot {
            empty(x.device(), Kind::Float)?
        } else {
            Tensor::f_empty([1, 1], (Kind::Float, x.device()))?
        };
        let (act_weight, act_code, act_eps) = if request.convrot {
            crate::convrot::activation_args(&x_2d, activation)?
        } else {
            (empty(x.device(), x.kind())?, 0, 0.)
        };
        let weight_scale = if request.weight_scale.device() == x.device()
            && request.weight_scale.kind() == Kind::Float
            && request.weight_scale.is_contiguous()
        {
            request.weight_scale.shallow_clone()
        } else {
            request
                .weight_scale
                .f_to_device_(x.device(), Kind::Float, false, false)?
                .f_reshape([-1])?
                .f_contiguous()?
        };
        let output = Tensor::f_empty([1, n], (request.output_kind, x.device()))?;
        let bias = match request.bias {
            Some(bias)
                if bias.device() != x.device()
                    || bias.kind() != request.output_kind
                    || !bias.is_contiguous() =>
            {
                bias.f_to_device_(x.device(), request.output_kind, false, false)?
                    .f_contiguous()?
            }
            Some(bias) => bias.shallow_clone(),
            None => empty(x.device(), request.output_kind)?,
        };
        let fused_residual = gemv_residual_args(native, request, n)?;
        fused_residual_done = fused_residual.is_some();
        unsafe {
            native.call_unchecked(
                "int8_linear_m1",
                &[
                    T(&x_2d),
                    T(&quantized),
                    T(&x_scale),
                    T(&weight),
                    T(&weight_scale),
                    T(&bias),
                    T(&output),
                    Int(output_code),
                    Bool(request.convrot),
                    Int(request.convrot_group),
                    Int(act_code),
                    T(&act_weight),
                    KernelArg::Float(act_eps),
                    Int(stream),
                    fused_residual
                        .as_ref()
                        .map_or(KernelArg::None, |(r, _)| T(r)),
                    fused_residual
                        .as_ref()
                        .map_or(KernelArg::None, |(_, s)| T(s)),
                ],
            )?;
        }
        output
    } else {
        let (quantized, x_scale) = if request.convrot {
            if fused {
                unsafe { crate::convrot::rowwise64(native, &x_2d, group, None, activation)? }
            } else if group == 256
                && k % 256 == 0
                && k <= 16384
                && (k <= 5120 || k >= 8192)
                && fused_shared_fits(native, device, m, k)?
            {
                unsafe { crate::convrot::rowwise(native, &x_2d, group, None)? }
            } else {
                let h = crate::int8_utils::hadamard(group, x_2d.device(), x_2d.kind())?;
                crate::convrot::quantize_activation(native, &x_2d, &h, group, None)?
            }
        } else {
            // The linear wrapper allocates directly; the standalone rowwise
            // helper's reshape(-1, K) would incorrectly reject the approved K=0.
            let q = Tensor::f_empty([m, k], (Kind::Int8, x.device()))?;
            let scale = Tensor::f_empty([m, 1], (Kind::Float, x.device()))?;
            unsafe {
                native.call_unchecked(
                    "quantize_int8_rowwise",
                    &[
                        T(&x_2d),
                        T(&q),
                        T(&scale),
                        KernelArg::Bool(false),
                        Int(0),
                        Int(stream),
                    ],
                )?;
            }
            (q, scale)
        };
        match route {
            LinearRoute::M1 | LinearRoute::Auto { .. } => unreachable!("route prepared above"),
            LinearRoute::Gemv => {
                if m == 2 && !aligned16(&weight) {
                    weight = native.clone_tensor(&weight)?;
                }
                let weight_scale = if request.weight_scale.device() == x.device()
                    && request.weight_scale.kind() == Kind::Float
                    && request.weight_scale.is_contiguous()
                {
                    request.weight_scale.shallow_clone()
                } else {
                    request
                        .weight_scale
                        .f_to_device_(x.device(), Kind::Float, false, false)?
                        .f_reshape([-1])?
                        .f_contiguous()?
                };
                let output = Tensor::f_empty([m, n], (request.output_kind, x.device()))?;
                let bias = match request.bias {
                    Some(bias)
                        if bias.device() != x.device()
                            || bias.kind() != request.output_kind
                            || !bias.is_contiguous() =>
                    {
                        bias.f_to_device_(x.device(), request.output_kind, false, false)?
                            .f_contiguous()?
                    }
                    Some(bias) => bias.shallow_clone(),
                    None => empty(x.device(), request.output_kind)?,
                };
                let fused_residual = if m == 1 {
                    gemv_residual_args(native, request, n)?
                } else {
                    None
                };
                fused_residual_done = fused_residual.is_some();
                unsafe {
                    native.call_unchecked(
                        "int8_gemv_dequant",
                        &[
                            T(&quantized),
                            T(&weight),
                            T(&x_scale),
                            T(&weight_scale),
                            T(&bias),
                            T(&output),
                            Int(output_code),
                            Int(stream),
                            fused_residual
                                .as_ref()
                                .map_or(KernelArg::None, |(r, _)| T(r)),
                            fused_residual
                                .as_ref()
                                .map_or(KernelArg::None, |(_, s)| T(s)),
                        ],
                    )?;
                }
                output
            }
            LinearRoute::Gemm { disable_cutlass } => {
                let output = Tensor::f_empty([m, n], (request.output_kind, x.device()))?;
                let weight_scale = if request.weight_scale.device() == x.device()
                    && request.weight_scale.kind() == Kind::Float
                    && request.weight_scale.is_contiguous()
                {
                    request.weight_scale.shallow_clone()
                } else {
                    request
                        .weight_scale
                        .f_to_device_(x.device(), Kind::Float, false, false)?
                        .f_reshape([-1])?
                        .f_contiguous()?
                };
                let bias = match request.bias {
                    Some(bias) => vector(native, bias, x.device(), request.output_kind)?,
                    None => empty(x.device(), request.output_kind)?,
                };
                let mut used_cutlass = false;
                // The frozen _prefer_cublas_int8_fallback returns false. SM89 is
                // not Turing; no SM75 preference/padding branch is selected here.
                if !disable_cutlass && supports_cutlass(native, device)? {
                    let ws_cutlass = if weight_scale.numel() == n as usize {
                        weight_scale.shallow_clone()
                    } else {
                        weight_scale.f_expand([n], false)?.f_contiguous()?
                    };
                    let residual_2d = match (request.residual, request.residual_scale) {
                        (Some(r), Some(scale))
                            if r.kind() == request.output_kind
                                && scale.kind() == request.output_kind =>
                        {
                            let r = r.f_reshape([m, n])?.f_contiguous()?;
                            if aligned16(&r) { Some(r) } else { None }
                        }
                        _ => None,
                    };
                    if let Some(r) = residual_2d {
                        let scale = vector(
                            native,
                            request.residual_scale.unwrap(),
                            x.device(),
                            request.output_kind,
                        )?;
                        used_cutlass = unsafe {
                            native.call_unchecked(
                                "cutlass_int8_dequant_residual",
                                &[
                                    T(&quantized),
                                    T(&weight),
                                    T(&x_scale),
                                    T(&ws_cutlass),
                                    T(&bias),
                                    T(&scale),
                                    T(&r),
                                    T(&output),
                                    Int(output_code),
                                    Int(stream),
                                ],
                            )?
                        }
                        .value
                            != 0;
                        fused_residual_done = used_cutlass;
                    }
                    if !used_cutlass {
                        used_cutlass = unsafe {
                            native.call_unchecked(
                                "cutlass_int8_dequant",
                                &[
                                    T(&quantized),
                                    T(&weight),
                                    T(&x_scale),
                                    T(&ws_cutlass),
                                    T(&bias),
                                    T(&output),
                                    Int(output_code),
                                    Int(stream),
                                ],
                            )?
                        }
                        .value
                            != 0;
                    }
                }
                if !used_cutlass {
                    dequantized_output = true;
                    let out_int32 = Tensor::f_empty([m, n], (Kind::Int, x.device()))?;
                    let workspace = cublas_workspace(native, device)?;
                    unsafe {
                        native.call_unchecked(
                            "cublas_gemm_int8",
                            &[
                                T(&quantized),
                                T(&weight),
                                T(&out_int32),
                                T(&workspace),
                                Int(stream),
                            ],
                        )?;
                        if let Some(observer) = observer.as_mut() {
                            observer(Int8Consumer::Dequantize, &[&out_int32, &x_scale])?;
                        }
                        native.call_unchecked(
                            "dequantize_int8_linear",
                            &[
                                T(&out_int32),
                                T(&x_scale),
                                T(&weight_scale),
                                T(&bias),
                                T(&output),
                                Int(output_code),
                                Int(stream),
                            ],
                        )?;
                    }
                }
                output
            }
        }
    };
    if k == 0
        && output.numel() > 0
        && !dequantized_output
        && let Some(observer) = observer.as_mut()
    {
        observer(Int8Consumer::UnwrittenOutput, &[&output])?;
    }
    let output = if fused_residual_done {
        output
    } else {
        let residual = request
            .residual
            .map(|r| r.f_reshape(output.size()))
            .transpose()?;
        int8_linear::apply_residual(&output, residual.as_ref(), request.residual_scale)?
    };
    if shape.len() == 2 {
        Ok(output)
    } else {
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(n);
        Ok(output.f_reshape(output_shape.as_slice())?)
    }
}

/// Original M=1 predicate for a plain prepared 2D input.
pub fn int8_m1_supported(
    native: &Kitchen,
    x_2d: &Tensor,
    weight: &Tensor,
    convrot: bool,
    group: i64,
) -> Result<bool, Error> {
    int8_m1_supported_width(native, x_2d, weight, convrot, group, x_2d.size()[1])
}
fn int8_m1_supported_width(
    native: &Kitchen,
    x_2d: &Tensor,
    weight: &Tensor,
    convrot: bool,
    group: i64,
    k: i64,
) -> Result<bool, Error> {
    let (m, n) = (x_2d.size()[0], weight.size()[0]);
    if m != 1 {
        return Ok(false);
    }
    if !convrot {
        return Ok(k % 4 == 0 && (k <= 2560 || (k == 6144 && n <= 128)));
    }
    if group != 256 || k % 256 != 0 || !(256..=16384).contains(&k) {
        return Ok(false);
    }
    let Device::Cuda(device) = x_2d.device() else {
        return Ok(true); // Original shared-memory helper accepts non-CUDA metadata.
    };
    Ok(aligned16(weight)
        && crate::convrot::fused_shared_memory_bytes(m, k) + k
            < crate::convrot::max_dynamic_shared_memory(native, device)?)
}

/// The original `int8_linear` M=1 branch, not a replacement for other routes.
/// # Safety
/// Caller must select the original M=1 route with input_act None, floating
/// CUDA input/output, 2D INT8 CUDA weight and valid original launch operands.
/// Zero-K unwritten bytes are only inspected under the separately approved
/// controlled-state criterion, never initialized here as a substitute result.
pub unsafe fn int8_m1_linear(native: &Kitchen, request: &Int8M1<'_>) -> Result<Tensor, Error> {
    unsafe {
        int8_dispatch(
            native,
            request,
            LinearRoute::M1,
            InputActivation::None,
            None,
        )
    }
}
