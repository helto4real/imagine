//! Frozen CUDA SVDQuant scaled-mm wrapper; executable backend binding is separate.
use crate::native::{Error, KernelArg, Kitchen};
use tch::{Device, Kind, Tensor};

/// Original per-source-tensor `_ck_natural_lora_up` attribute. Keep this object
/// with the source tensor; a second Python-style tensor object has its own cache.
pub struct LoraUp {
    pub tensor: Tensor,
    natural: Option<Tensor>,
}
impl LoraUp {
    pub fn cached_natural(&self) -> Option<&Tensor> {
        self.natural.as_ref()
    }
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            natural: None,
        }
    }
    pub fn natural(&mut self) -> Result<Tensor, Error> {
        if self.tensor.dim() != 3 {
            return Ok(self.tensor.shallow_clone());
        }
        let shape = self.tensor.size();
        let expected = [shape[0] * 128, shape[1]];
        if let Some(cached) = &self.natural
            && cached.device() == self.tensor.device()
            && cached.kind() == self.tensor.kind()
            && cached.size() == expected
        {
            return Ok(cached.shallow_clone());
        }
        let natural = self
            .tensor
            .f_permute([0, 2, 1])?
            .f_reshape(expected)?
            .f_contiguous()?;
        self.natural = Some(natural.shallow_clone());
        Ok(natural)
    }
}

/// Values of the original three module environment overrides. None means unset.
#[derive(Default)]
pub struct Settings {
    pub fast_accum: Option<String>,
    pub shared_scale: Option<String>,
    pub fuse_lora_up: Option<String>,
}
fn enabled(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchOptions {
    pub act_unsigned: bool,
    pub fast_accum: bool,
    pub shared_scale: bool,
    pub fuse_lora: bool,
}
pub struct ScaledMm<'a> {
    pub act: &'a Tensor,
    pub weight: &'a Tensor,
    pub act_scales: &'a Tensor,
    pub weight_scales: &'a Tensor,
    pub lora_act: &'a Tensor,
    pub lora_up: &'a mut LoraUp,
    pub bias: Option<&'a Tensor>,
    pub act_unsigned: bool,
}
pub type OutputObserver<'a> = dyn FnMut(&Tensor, LaunchOptions) -> Result<(), Error> + 'a;

/// # Safety
/// All operands must meet the original selected CUDA native layout, capacity,
/// dtype and current-device contracts. No new input rejection is introduced.
pub unsafe fn scaled_mm(
    native: &Kitchen,
    request: &mut ScaledMm<'_>,
    settings: &Settings,
) -> Result<Tensor, Error> {
    unsafe { scaled_mm_impl(native, request, settings, None) }
}

/// Metadata-only development observer before the original unfused ATen consumer
/// or result return. Ordinary calls install none. No tensor lifetime is retained.
/// # Safety
/// Same contract as scaled_mm; an observer must not read/fill hidden storage or
/// change stream state, and must refuse unknown no-write ranges before any read.
/// D-05-SVDQUANT-NONGROUP-OUTPUT governs the approved eight controlled cases.
#[doc(hidden)]
pub unsafe fn scaled_mm_observed(
    native: &Kitchen,
    request: &mut ScaledMm<'_>,
    settings: &Settings,
    observer: &mut OutputObserver<'_>,
) -> Result<Tensor, Error> {
    unsafe { scaled_mm_impl(native, request, settings, Some(observer)) }
}
unsafe fn scaled_mm_impl(
    native: &Kitchen,
    request: &mut ScaledMm<'_>,
    settings: &Settings,
    observer: Option<&mut OutputObserver<'_>>,
) -> Result<Tensor, Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let m = request.act.size()[0];
    let packed = request.weight.dim() == 4;
    let n = request.weight.size()[0] * if packed { 128 } else { 1 };
    let kind = request.lora_up.tensor.kind();
    let mut out = Tensor::f_empty([m, n], (kind, request.act.device()))?;
    let empty = Tensor::f_empty([0], (kind, request.act.device()))?;
    let options = LaunchOptions {
        act_unsigned: request.act_unsigned,
        fast_accum: settings.fast_accum.as_deref().is_some_and(enabled),
        shared_scale: settings.shared_scale.as_deref().map_or(packed, enabled),
        fuse_lora: request.lora_act.kind() == kind
            && packed == (request.lora_up.tensor.dim() == 3)
            && settings.fuse_lora_up.as_deref().is_none_or(enabled),
    };
    let Device::Cuda(device) = request.act.device() else {
        return Err(Error::Native("SVDQuant requires CUDA input".into()));
    };
    let stream = native.current_stream(device)? as i64;
    let act = request.act.f_view_dtype(Kind::Uint8)?;
    let weight = request.weight.f_view_dtype(Kind::Uint8)?;
    unsafe {
        native.call_unchecked(
            "svdquant_scaled_mm_w4a4",
            &[
                T(&act),
                T(&weight),
                T(request.act_scales),
                T(request.weight_scales),
                T(request.lora_act),
                T(&request.lora_up.tensor),
                T(request.bias.unwrap_or(&empty)),
                T(&out),
                Bool(options.act_unsigned),
                Bool(options.fast_accum),
                Bool(options.shared_scale),
                Bool(options.fuse_lora),
                Int(stream),
            ],
        )?;
    }
    if let Some(callback) = observer {
        callback(&out, options)?;
    }
    if !options.fuse_lora {
        let lora = if request.lora_act.kind() == kind {
            request.lora_act.shallow_clone()
        } else {
            request.lora_act.f_to_kind(kind)?
        };
        let natural = request.lora_up.natural()?;
        let _ = out.f_addmm_(&lora, &natural.f_transpose(0, 1)?)?;
    }
    Ok(out)
}

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
type WorkspaceKey = (usize, u64, Kind, Kind, i64, i64, i64);
type Workspace = (Tensor, Tensor, Tensor);
static WORKSPACES: LazyLock<Mutex<HashMap<WorkspaceKey, Workspace>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Default)]
pub struct QuantizeSettings {
    pub lora_act_fp32: Option<String>,
    pub reuse_workspace: Option<String>,
}
pub struct Quantize<'a> {
    pub x: &'a Tensor,
    pub smooth: &'a Tensor,
    pub lora_down: &'a Tensor,
    pub pad_size: i64,
    pub act_unsigned: bool,
    pub lora_x: Option<&'a Tensor>,
}

/// Original smooth/INT4 quantization, per-stream workspace and ATen LoRA-down.
/// # Safety
/// Selected CUDA input/smooth/LoRA operands must satisfy original dtype/layout
/// and native capacity requirements. The original explicit assertions remain.
pub unsafe fn quantize(
    native: &Kitchen,
    request: &Quantize<'_>,
    settings: &QuantizeSettings,
) -> Result<(Tensor, Tensor, Tensor), Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let x = request.x;
    if x.dim() != 2 || !x.is_contiguous() {
        return Err(Error::ReferenceException {
            class: "AssertionError",
            message: "x must be 2D contiguous".into(),
        });
    }
    let (m, k) = (x.size()[0], x.size()[1]);
    let r = request.lora_down.size()[1];
    if request.pad_size == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "integer division or modulo by zero".into(),
        });
    }
    // Preserve Python floor division, including its negative-divisor rule.
    let numerator = m + request.pad_size - 1;
    let mut quotient = numerator / request.pad_size;
    if numerator % request.pad_size != 0 && (numerator < 0) != (request.pad_size < 0) {
        quotient -= 1;
    }
    let padded = quotient * request.pad_size;
    if k % 64 != 0 {
        return Err(Error::ReferenceException {
            class: "AssertionError",
            message: format!("K={k} must be divisible by group_size=64"),
        });
    }
    let lora_kind = if settings.lora_act_fp32.as_deref().is_some_and(enabled) {
        Kind::Float
    } else {
        x.kind()
    };
    let Device::Cuda(device) = x.device() else {
        return Err(Error::Native("SVDQuant requires CUDA input".into()));
    };
    let stream = native.current_stream(device)?;
    let reuse = !settings
        .reuse_workspace
        .as_deref()
        .is_some_and(|v| matches!(v.to_lowercase().as_str(), "0" | "false" | "no" | "off"));
    let allocate = || -> Result<Workspace, Error> {
        Ok((
            Tensor::f_empty([padded, k / 2], (Kind::Uint8, x.device()))?,
            Tensor::f_empty([k / 64, padded], (x.kind(), x.device()))?,
            Tensor::f_empty([padded, r], (lora_kind, x.device()))?,
        ))
    };
    let (q, scales, lora) = if reuse {
        let key = (device, stream, x.kind(), lora_kind, padded, k, r);
        let mut cache = WORKSPACES.lock().unwrap();
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(key) {
            entry.insert(allocate()?);
        }
        let (q, s, l) = &cache[&key];
        (q.shallow_clone(), s.shallow_clone(), l.shallow_clone())
    } else {
        allocate()?
    };
    unsafe {
        native.call_unchecked(
            "svdquant_quantize_w4a4",
            &[
                T(x),
                T(request.smooth),
                T(request.lora_down),
                T(&q),
                T(&scales),
                T(&lora),
                Bool(request.act_unsigned),
                Int(stream as i64),
            ],
        )?;
    }
    let source = request.lora_x.unwrap_or(x);
    if m > 0 {
        let mut rows = lora.f_narrow(0, 0, m)?;
        if rows.kind() == source.kind() && rows.is_contiguous() {
            let _ = source.f_mm_out(&rows, request.lora_down)?;
        } else {
            let product = source.f_matmul(request.lora_down)?;
            native.copy_tensor(&mut rows, &product, true)?;
        }
    }
    if padded > m {
        let _ = lora.f_narrow(0, m, padded - m)?.f_zero_()?;
    }
    Ok((q.f_view_dtype(Kind::Int8)?, scales, lora))
}
