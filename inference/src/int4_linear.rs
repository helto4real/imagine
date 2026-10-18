//! Frozen typed signed INT4 selection on initial SM89; backend binding is separate.
use crate::{
    convrot_w4a4, cuda_linear,
    native::{Error, KernelArg, Kitchen},
};
use tch::{Device, Kind, Tensor};

pub struct Linear<'a> {
    pub activation: &'a Tensor,
    pub weight: &'a Tensor,
    pub activation_scale: &'a Tensor,
    pub weight_scale: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub output_kind: Kind,
}
#[derive(Clone, Copy)]
pub struct Settings {
    pub force_int8_fallback: bool,
    pub disable_cutlass: bool,
    /// Original parsed/clamped COMFY_KITCHEN_INT4_INT8_WEIGHT_CHUNK_N value.
    pub weight_chunk_n: i64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            force_int8_fallback: false,
            disable_cutlass: false,
            weight_chunk_n: 4096,
        }
    }
}
#[derive(Clone, Debug)]
pub struct NativeCall {
    pub name: &'static str,
    pub returned: Option<bool>,
    pub chunk_cols: Option<i64>,
    pub allow_sm80_cutlass: Option<bool>,
}
impl NativeCall {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            returned: None,
            chunk_cols: None,
            allow_sm80_cutlass: None,
        }
    }
}
pub type TraceObserver<'a> = dyn FnMut(&Tensor, &[NativeCall]) -> Result<(), Error> + 'a;

/// Original legacy preference, with the original per-device SM count.
pub fn prefers_legacy(m: i64, n: i64, k: i64, sm_count: i64) -> bool {
    let tile = (m.max(1) as u64).next_power_of_two().max(32) as i128;
    let base = if k <= 1024 {
        1_048_576_i128
    } else {
        786_432_i128
    };
    let threshold = base * sm_count as i128 / 142;
    let workload = tile * n as i128;
    n < 32768
        && (workload < threshold
            || (k > 1024 && workload == threshold && m >= 128 && m as i128 == tile && m < 512))
}
/// # Safety
/// Operands must satisfy the original selected CUDA native layout/capacity and
/// active-device contracts on SM89, with original FP32/FP16/BF16 output kind.
pub unsafe fn linear(
    native: &Kitchen,
    request: &Linear<'_>,
    settings: &Settings,
) -> Result<Tensor, Error> {
    unsafe { linear_impl(native, request, settings, None) }
}
/// Metadata-only development trace before result return. Ordinary calls install
/// none. Guard all no-write output ranges before any inspection/consumption.
/// # Safety
/// Same contract as linear. Callback must not read/fill hidden bytes, change
/// streams or retain extra tensors; it must refuse unmatched no-write storage.
#[doc(hidden)]
pub unsafe fn linear_observed(
    native: &Kitchen,
    request: &Linear<'_>,
    settings: &Settings,
    observer: &mut TraceObserver<'_>,
) -> Result<Tensor, Error> {
    unsafe { linear_impl(native, request, settings, Some(observer)) }
}

pub type OutputObserver<'a> = dyn FnMut(&Tensor) -> Result<(), Error> + 'a;
fn value_error(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}

/// Original int4_linear prefix and its selected forced-INT8 packed-GEMV route.
/// # Safety
/// Caller must select the original force-INT8 option and M<=8 on SM89. Packed
/// INT8 operands must satisfy the original native CUDA capacity/stream contract;
/// output must be FP32/FP16/BF16. This helper does not install an observer.
pub unsafe fn forced_small_m(native: &Kitchen, request: &Linear<'_>) -> Result<Tensor, Error> {
    unsafe {
        linear_impl(
            native,
            request,
            &Settings {
                force_int8_fallback: true,
                ..Settings::default()
            },
            None,
        )
    }
}

/// Metadata-only development hook before result return; no observer in ordinary
/// calls. D-05-INT4-FORCED-FALLBACK-ZERO-K-OUTPUT bounds no-write inspection.
/// # Safety
/// Same contract as forced_small_m. Observer must refuse unmatched complete
/// ranges before reading/consumption, never read/fill hidden bytes or change
/// stream state, and retain no extra tensor lifetime.
#[doc(hidden)]
pub unsafe fn forced_small_m_observed(
    native: &Kitchen,
    request: &Linear<'_>,
    observer: &mut OutputObserver<'_>,
) -> Result<Tensor, Error> {
    let mut bridge = |output: &Tensor, _calls: &[NativeCall]| observer(output);
    unsafe {
        linear_impl(
            native,
            request,
            &Settings {
                force_int8_fallback: true,
                ..Settings::default()
            },
            Some(&mut bridge),
        )
    }
}
unsafe fn linear_impl(
    native: &Kitchen,
    request: &Linear<'_>,
    settings: &Settings,
    observer: Option<&mut TraceObserver<'_>>,
) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    let x = request.activation;
    let w = request.weight;
    if x.dim() != 2 || w.dim() != 2 {
        return Err(value_error(
            "INT4 linear expects 2D activation and weight tensors",
        ));
    }
    if x.size()[1] != w.size()[1] {
        return Err(value_error(
            "INT4 linear activation/weight K dimensions do not match",
        ));
    }
    let m = x.size()[0];
    let n = w.size()[0];
    // Original outer output remains alive throughout the nested fallback call.
    let _outer_output = Tensor::f_empty([m, n], (request.output_kind, x.device()))?;
    let xs = request
        .activation_scale
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_reshape([-1])?
        .f_contiguous()?;
    let ws = request
        .weight_scale
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_reshape([-1])?
        .f_contiguous()?;
    if xs.numel() != m as usize {
        return Err(value_error(format!(
            "INT4 x_scale must have {m} values, got {}",
            xs.numel()
        )));
    }
    if ws.numel() != n as usize {
        return Err(value_error(format!(
            "INT4 weight_scale must have {n} values, got {}",
            ws.numel()
        )));
    }
    let bias = match request.bias {
        Some(b) => cuda_linear::vector(native, b, x.device(), request.output_kind)?,
        None => cuda_linear::empty(x.device(), request.output_kind)?,
    };
    let Device::Cuda(device) = x.device() else {
        return Err(Error::Native("CUDA INT4 requires CUDA input".into()));
    };
    let outer_stream = native.current_stream(device)?;
    let k = x.size()[1] * 2;
    let code = match request.output_kind {
        Kind::Float => 0,
        Kind::Half => 1,
        Kind::BFloat16 => 2,
        _ => unreachable!("Original selected output dtype contract"),
    };
    let mut calls = Vec::new();
    if !settings.force_int8_fallback {
        let mut used = false;
        if !settings.disable_cutlass
            && !prefers_legacy(
                m,
                n,
                k,
                native.device_properties(device)?.multiprocessor_count,
            )
            && cuda_linear::supports_cutlass(native, device)?
        {
            used = unsafe {
                native.call_unchecked(
                    "cutlass_int4_dequant",
                    &[
                        T(&x.f_contiguous()?),
                        T(&w.f_contiguous()?),
                        T(&xs),
                        T(&ws),
                        T(&bias),
                        T(&_outer_output),
                        Int(code),
                        Int(outer_stream as i64),
                    ],
                )?
            }
            .value
                != 0;
            let mut event = NativeCall::new("cutlass_int4_dequant");
            event.returned = Some(used);
            calls.push(event);
        }
        if !used {
            unsafe {
                native.call_unchecked(
                    "int4_linear",
                    &[
                        T(&x.f_contiguous()?),
                        T(&w.f_contiguous()?),
                        T(&xs),
                        T(&ws),
                        T(&bias),
                        T(&_outer_output),
                        Int(code),
                        Int(outer_stream as i64),
                    ],
                )?;
            }
            calls.push(NativeCall::new("int4_linear"));
        }
        if let Some(callback) = observer {
            callback(&_outer_output, &calls)?;
        }
        return Ok(_outer_output);
    }
    let x = x.f_contiguous()?;
    let w = w.f_contiguous()?;
    let unpacked = unsafe { convrot_w4a4::prepare_int8_weight(native, &x)? };
    calls.push(NativeCall::new("unpack_int4_to_int8"));
    let chunk_cols = if n <= 2560 {
        n
    } else {
        n.min(settings.weight_chunk_n.max(1))
    };
    let output = Tensor::f_empty([m, n], (request.output_kind, x.device()))?;
    let workspaces = if m > 8 {
        Some((
            Tensor::f_empty([chunk_cols, k], (Kind::Int8, x.device()))?,
            Tensor::f_empty([m, chunk_cols], (Kind::Int, x.device()))?,
        ))
    } else {
        None
    };
    let xs = xs
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_reshape([-1, 1])?
        .f_contiguous()?;
    if xs.size()[0] != m {
        return Err(value_error(format!(
            "{} INT4 weight {} x_scale must have {m} values, got {}",
            if m <= 8 { "packed" } else { "chunked" },
            if m <= 8 { "GEMV" } else { "GEMM" },
            xs.size()[0]
        )));
    }
    let ws = ws
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_reshape([-1])?
        .f_contiguous()?;
    if ws.numel() != n as usize {
        return Err(value_error(format!(
            "{} INT4 weight {} weight_scale must have {n} values, got {}",
            if m <= 8 { "packed" } else { "chunked" },
            if m <= 8 { "GEMV" } else { "GEMM" },
            ws.numel()
        )));
    }
    let stream = native.current_stream(device)? as i64;
    if let Some((weight_workspace, acc_workspace)) = &workspaces {
        let workspace = cuda_linear::cublas_workspace(native, device)?;
        unsafe {
            native.call_unchecked(
                "int4_weight_int8_act_gemm_dequant_chunked",
                &[
                    T(&unpacked),
                    T(&w),
                    T(&xs),
                    T(&ws),
                    T(&bias),
                    T(&output),
                    T(weight_workspace),
                    T(acc_workspace),
                    T(&workspace),
                    Int(chunk_cols),
                    KernelArg::Bool(true),
                    Int(code),
                    Int(stream),
                ],
            )?;
        }
        let mut event = NativeCall::new("int4_weight_int8_act_gemm_dequant_chunked");
        event.chunk_cols = Some(chunk_cols);
        event.allow_sm80_cutlass = Some(true);
        calls.push(event);
    } else {
        unsafe {
            native.call_unchecked(
                "int4_weight_int8_act_gemv_dequant",
                &[
                    T(&unpacked),
                    T(&w.f_contiguous()?),
                    T(&xs),
                    T(&ws),
                    T(&bias),
                    T(&output),
                    Int(code),
                    Int(stream),
                ],
            )?;
        }
        calls.push(NativeCall::new("int4_weight_int8_act_gemv_dequant"));
    }
    if let Some(callback) = observer {
        callback(&output, &calls)?;
    }
    Ok(output)
}
