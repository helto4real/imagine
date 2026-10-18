//! Frozen CUDA Sage-attention validation, packing, mask preparation and launch.
use crate::{
    execution::Runtime,
    native::{Error, KernelArg},
    registry::AllowedType,
};
use tch::{Device, Kind, Tensor};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn shape(s: &[i64]) -> String {
    format!(
        "({}{})",
        s.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
        if s.len() == 1 { "," } else { "" }
    )
}
fn dtype(k: Kind) -> &'static str {
    AllowedType::DType(k).python_name()
}
fn code(k: Kind) -> Result<i64, Error> {
    match k {
        Kind::Float => Ok(0),
        Kind::Half => Ok(1),
        Kind::BFloat16 => Ok(2),
        Kind::Float8e4m3fn => Ok(5),
        Kind::Float8e5m2 => Ok(6),
        _ => Err(error("KeyError", dtype(k))),
    }
}
fn dimensions(t: &Tensor) -> Result<[i64; 4], Error> {
    let s = t.size();
    s.as_slice().try_into().map_err(|_| {
        error(
            "ValueError",
            if s.len() < 4 {
                format!("not enough values to unpack (expected 4, got {})", s.len())
            } else {
                "too many values to unpack (expected 4)".into()
            },
        )
    })
}
/// Owned packed tensors only. Floating inputs are not retained after quantization.
pub struct PrequantizedInt8Attention {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub q_scale: Tensor,
    pub k_scale: Tensor,
    pub v_scale: Tensor,
    pub original_head_dim: i64,
    pub input_dtype: Kind,
    pub attention_scale: f64,
    pub cta_k: i64,
    pub attn_mask: Option<Tensor>,
}
/// Development-only known-own-allocation provenance; ordinary calls install none.
pub trait MaskAllocationObserver {
    fn before(&mut self, shape: [i64; 3], device: Device) -> Result<(), Error>;
    fn after(&mut self, tensor: &Tensor) -> Result<(), Error>;
}
struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    packed: PrequantizedInt8Attention,
}
fn validate(
    rt: &Runtime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Option<Tensor>, Error> {
    if q.dim() != 4 || k.dim() != 4 || v.dim() != 4 {
        return Err(error(
            "ValueError",
            "q, k, and v must have shape [batch, heads, sequence, head_dim]",
        ));
    }
    if !matches!(q.kind(), Kind::Float | Kind::Half | Kind::BFloat16) {
        return Err(error(
            "TypeError",
            format!(
                "q, k, and v must be float32, float16, or bfloat16, got {}",
                dtype(q.kind())
            ),
        ));
    }
    if q.kind() != k.kind() || q.kind() != v.kind() {
        return Err(error(
            "TypeError",
            format!(
                "q, k, and v must have the same dtype, got {}, {}, and {}",
                dtype(q.kind()),
                dtype(k.kind()),
                dtype(v.kind())
            ),
        ));
    }
    if !matches!(q.device(), Device::Cuda(_))
        || q.device() != k.device()
        || q.device() != v.device()
    {
        return Err(error(
            "ValueError",
            "q, k, and v must be on the same CUDA device",
        ));
    }
    if !rt.int8_attention_available(Some(q.device()))? {
        return Err(unavailable());
    }
    let [b, h, l, d] = dimensions(q)?;
    let [kb, kh, kl, kd] = dimensions(k)?;
    if v.size() != [b, kh, kl, d] {
        return Err(error(
            "ValueError",
            format!(
                "v must have shape [q.batch, k.heads, k.sequence, q.head_dim], got {}",
                shape(&v.size())
            ),
        ));
    }
    if kb != b || kd != d {
        return Err(error(
            "ValueError",
            format!(
                "q and k batch/head dimensions must match, got {} and {}",
                shape(&q.size()),
                shape(&k.size())
            ),
        ));
    }
    if b == 0 || h == 0 || kh == 0 || l == 0 || kl == 0 {
        return Err(error(
            "ValueError",
            "batch, head counts, and sequence lengths must be positive",
        ));
    }
    if h % kh != 0 {
        return Err(error(
            "ValueError",
            format!("q head count ({h}) must be divisible by k/v head count ({kh})"),
        ));
    }
    if d <= 0 || d > 256 {
        return Err(error(
            "ValueError",
            format!("head_dim must be in [1, 256], got {d}"),
        ));
    }
    if q.stride()[3] != 1 || k.stride()[3] != 1 || v.stride()[3] != 1 {
        return Err(error(
            "ValueError",
            "the last dimension of q, k, and v must be contiguous",
        ));
    }
    let Some(mask) = mask else { return Ok(None) };
    if mask.device() != q.device() {
        return Err(error(
            "ValueError",
            "attn_mask must be on the same CUDA device as q, k, and v",
        ));
    }
    if !matches!(
        mask.kind(),
        Kind::Bool | Kind::Half | Kind::BFloat16 | Kind::Float
    ) {
        return Err(error(
            "TypeError",
            "attn_mask must be bool, float16, bfloat16, or float32",
        ));
    }
    mask.f_broadcast_to([b, h, l, kl]).map(Some).map_err(|_| {
        error(
            "ValueError",
            format!(
                "attn_mask must be broadcastable to [{b}, {h}, {l}, {kl}], got {}",
                shape(&mask.size())
            ),
        )
    })
}
fn unavailable() -> Error {
    error(
        "RuntimeError",
        "INT8 attention requires the comfy-kitchen CUDA extension on SM75 or newer, or the HIP extension on an AMD device with matrix cores (RDNA3 or newer)",
    )
}
fn stream(rt: &Runtime, t: &Tensor) -> Result<i64, Error> {
    let Device::Cuda(i) = t.device() else {
        unreachable!()
    };
    Ok(rt.native.current_stream(i)? as i64)
}
unsafe fn mask(
    rt: &Runtime,
    mask: Option<Tensor>,
    scale: f64,
    observer: &mut Option<&mut dyn MaskAllocationObserver>,
) -> Result<Option<Tensor>, Error> {
    let Some(mask) = mask else { return Ok(None) };
    if mask.size()[3] <= 1024 || mask.stride()[2] != 0 || scale <= 0. {
        return Ok(Some(mask));
    }
    let mb = if mask.stride()[0] == 0 {
        1
    } else {
        mask.size()[0]
    };
    let mh = if mask.stride()[1] == 0 {
        1
    } else {
        mask.size()[1]
    };
    let len = mask.size()[3];
    let compact = mask
        .f_slice(0, 0, mb, 1)?
        .f_slice(1, 0, mh, 1)?
        .f_slice(2, 0, 1, 1)?;
    let tiles = (len + 127) / 128;
    let width = ((tiles * 129 + 3) / 4) * 4;
    let shape = [mb, mh, width];
    if let Some(observer) = observer.as_deref_mut() {
        observer.before(shape, mask.device())?;
    }
    let packed = Tensor::f_empty(shape, (Kind::Float, mask.device()))?;
    if let Some(observer) = observer.as_deref_mut() {
        observer.after(&packed)?;
    }
    use KernelArg::{Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            "sage_prepare_key_mask",
            &[T(&compact), T(&packed), Int(stream(rt, &mask)?)],
        )?;
    }
    Ok(Some(packed))
}
unsafe fn prepare(
    rt: &Runtime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<f64>,
    attn_mask: Option<&Tensor>,
    mut observer: Option<&mut dyn MaskAllocationObserver>,
) -> Result<Inputs, Error> {
    let mask = validate(rt, q, k, v, attn_mask)?;
    let original = q.size()[3];
    let head = if original <= 64 {
        64
    } else if original <= 128 {
        128
    } else {
        256
    };
    let pad = |x: &Tensor| -> Result<Tensor, Error> {
        Ok(if head == original {
            x.shallow_clone()
        } else {
            x.f_pad([0, head - original], "constant", None)?
        })
    };
    let q = pad(q)?;
    let k = pad(k)?;
    let v = pad(v)?;
    let scale = scale.unwrap_or((original as f64).powf(-0.5));
    if !scale.is_finite() {
        return Err(error(
            "ValueError",
            format!(
                "scale must be finite, got {}",
                if scale.is_nan() {
                    "nan".into()
                } else {
                    scale.to_string()
                }
            ),
        ));
    }
    let mask = unsafe { self::mask(rt, mask, scale, &mut observer)? };
    let [b, h, l, _] = dimensions(&q)?;
    let [_, kh, kl, _] = dimensions(&k)?;
    let cta = if mask.as_ref().is_none_or(|m| m.dim() != 4) && head >= 128 && kl > 1024 {
        128
    } else {
        64
    };
    let padded = ((kl + cta - 1) / cta) * cta;
    let packed = PrequantizedInt8Attention {
        q: Tensor::f_empty(q.size(), (Kind::Int8, q.device()))?,
        k: Tensor::f_empty(k.size(), (Kind::Int8, k.device()))?,
        q_scale: Tensor::f_empty(
            [b, h, ((l + 127) / 128) * if head == 256 { 64 } else { 32 }],
            (Kind::Float, q.device()),
        )?,
        k_scale: Tensor::f_empty(
            [b, kh, ((kl + cta - 1) / cta) * 4],
            (Kind::Float, q.device()),
        )?,
        v: Tensor::f_empty([b * kh * head, padded], (Kind::Int8, q.device()))?,
        v_scale: Tensor::f_empty([b * kh * head], (Kind::Float, q.device()))?,
        original_head_dim: original,
        input_dtype: q.kind(),
        attention_scale: scale,
        cta_k: cta,
        attn_mask: mask,
    };
    Ok(Inputs { q, k, v, packed })
}
impl Runtime {
    pub fn int8_attention_available(&self, device: Option<Device>) -> Result<bool, Error> {
        if device.is_some_and(|d| !matches!(d, Device::Cuda(_))) || !self.native.cuda_available()? {
            return Ok(false);
        }
        let device = match device {
            Some(Device::Cuda(i)) => i,
            _ => self.native.current_device()?,
        };
        let p = self.native.device_properties(device)?;
        Ok((p.major, p.minor) >= (7, 5))
    }
    /// # Safety
    /// Original raw CUDA metadata, capacities and current-stream requirements.
    pub unsafe fn prequantize_int8_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: Option<f64>,
        mask: Option<&Tensor>,
    ) -> Result<PrequantizedInt8Attention, Error> {
        unsafe { self.prequantize_int8_attention_observed(q, k, v, scale, mask, None) }
    }
    /// # Safety
    /// Same original launch contracts. Observer may only prime/prove own test history.
    #[doc(hidden)]
    pub unsafe fn prequantize_int8_attention_observed(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: Option<f64>,
        mask: Option<&Tensor>,
        observer: Option<&mut dyn MaskAllocationObserver>,
    ) -> Result<PrequantizedInt8Attention, Error> {
        let input = unsafe { prepare(self, q, k, v, scale, mask, observer)? };
        let p = &input.packed;
        let [b, _, _, _] = dimensions(&input.q)?;
        let kh = input.k.size()[1];
        let anchor = Tensor::f_empty([b, kh], (Kind::Int, input.q.device()))?;
        use KernelArg::{Int, Tensor as T};
        unsafe {
            self.native.call_unchecked(
                "sage_sdpa_quantize",
                &[
                    T(&input.q),
                    T(&input.k),
                    T(&input.v),
                    T(&p.q),
                    T(&p.q_scale),
                    T(&p.k),
                    T(&p.k_scale),
                    T(&p.v),
                    T(&p.v_scale),
                    Int(p.cta_k),
                    Int(code(p.input_dtype)?),
                    Int(stream(self, &input.q)?),
                    Int(anchor.data_ptr() as i64),
                ],
            )?;
        }
        Ok(input.packed)
    }
    /// # Safety
    /// Original packed tensor metadata, capacities and stream ordering.
    pub unsafe fn int8_attention_from_prequantized(
        &self,
        p: &PrequantizedInt8Attention,
    ) -> Result<Tensor, Error> {
        if !matches!(p.q.device(), Device::Cuda(_)) {
            return Err(error(
                "ValueError",
                "prequantized INT8 attention tensors must be on a CUDA device",
            ));
        }
        if [&p.k, &p.v, &p.q_scale, &p.k_scale, &p.v_scale]
            .iter()
            .any(|t| t.device() != p.q.device())
        {
            return Err(error(
                "ValueError",
                "prequantized INT8 attention tensors must be on the same CUDA device",
            ));
        }
        if p.attn_mask
            .as_ref()
            .is_some_and(|m| m.device() != p.q.device())
        {
            return Err(error(
                "ValueError",
                "attn_mask must be on the same CUDA device as the packed tensors",
            ));
        }
        if !self.int8_attention_available(Some(p.q.device()))? {
            return Err(unavailable());
        }
        let [b, h, l, d] = dimensions(&p.q)?;
        let kind = if p.input_dtype == Kind::Float {
            Kind::BFloat16
        } else {
            p.input_dtype
        };
        let out = Tensor::f_empty([b, h, l, d], (kind, p.q.device()))?;
        use KernelArg::{Float, Int, Tensor as T};
        unsafe {
            self.native.call_unchecked(
                "sage_sdpa_prequantized",
                &[
                    T(&p.q),
                    T(&p.k),
                    T(&p.v),
                    T(&out),
                    T(&p.q_scale),
                    T(&p.k_scale),
                    T(&p.v_scale),
                    Int(p.cta_k),
                    Float(p.attention_scale),
                    Int(code(kind)?),
                    Int(stream(self, &p.q)?),
                    p.attn_mask.as_ref().map_or(KernelArg::None, T),
                ],
            )?;
        }
        let out = out.f_slice(-1, 0, p.original_head_dim, 1)?;
        Ok(if p.input_dtype == Kind::Float {
            out.f_to_kind(Kind::Float)?
        } else {
            out
        })
    }
    /// # Safety
    /// Same source CUDA input capacities and stream requirements.
    pub unsafe fn int8_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: Option<f64>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        unsafe { self.int8_attention_observed(q, k, v, scale, mask, None) }
    }
    /// # Safety
    /// Original launch contracts; observer is limited to known own test history.
    #[doc(hidden)]
    pub unsafe fn int8_attention_observed(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: Option<f64>,
        mask: Option<&Tensor>,
        observer: Option<&mut dyn MaskAllocationObserver>,
    ) -> Result<Tensor, Error> {
        let input = unsafe { prepare(self, q, k, v, scale, mask, observer)? };
        let p = &input.packed;
        let kind = if p.input_dtype == Kind::Float {
            Kind::BFloat16
        } else {
            p.input_dtype
        };
        let out = Tensor::f_empty(input.q.size(), (kind, input.q.device()))?;
        let [b, _, _, _] = dimensions(&input.q)?;
        let kh = input.k.size()[1];
        let anchor = Tensor::f_empty([b, kh], (Kind::Int, input.q.device()))?;
        use KernelArg::{Float, Int, Tensor as T};
        unsafe {
            self.native.call_unchecked(
                "sage_sdpa",
                &[
                    T(&input.q),
                    T(&input.k),
                    T(&input.v),
                    T(&out),
                    T(&p.q),
                    T(&p.q_scale),
                    T(&p.k),
                    T(&p.k_scale),
                    T(&p.v),
                    T(&p.v_scale),
                    Float(p.attention_scale),
                    Int(code(p.input_dtype)?),
                    Int(code(kind)?),
                    Int(stream(self, &input.q)?),
                    Int(anchor.data_ptr() as i64),
                    p.attn_mask.as_ref().map_or(KernelArg::None, T),
                    Int(p.cta_k),
                ],
            )?;
        }
        let out = out.f_slice(-1, 0, p.original_head_dim, 1)?;
        Ok(if p.input_dtype == Kind::Float {
            out.f_to_kind(Kind::Float)?
        } else {
            out
        })
    }
}
