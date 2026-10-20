//! Frozen W4A8/W6A8 validation, INT8-grid decode, inverse rotation and eager linear.
use super::*;
use crate::{convrot, packing};

pub(super) fn err(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(super) fn tuple(s: &[i64]) -> String {
    format!(
        "({}{})",
        s.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
        if s.len() == 1 { "," } else { "" }
    )
}
pub(crate) struct Weight {
    pub(crate) q: Tensor,
    pub(crate) rel: Tensor,
    pub(crate) channel: Tensor,
    pub(crate) codebook: Option<Tensor>,
    pub(crate) correction: Option<Tensor>,
    pub(crate) group: i64,
    pub(crate) rotation: i64,
    pub(crate) geometry: (i64, i64, i64),
}
impl Weight {
    pub(crate) fn parse(a: &[Value], k: &Kwargs, linear: bool) -> Result<Self, Error> {
        let i = usize::from(linear);
        let q = tensor(a, k, i, "qdata")?;
        let rel = tensor(a, k, i + 1, "s_rel")?;
        let channel = tensor(a, k, i + 2, "s_channel")?;
        let codebook = optional_tensor(a, k, i + 3, "codebook")?;
        let correction = optional_tensor(a, k, i + 4, "correction")?;
        let group = integer(get(a, k, if linear { 7 } else { 5 }, "group_size"), 16)?;
        let rotation = integer(
            get(a, k, if linear { 8 } else { 6 }, "convrot_groupsize"),
            256,
        )?;
        if q.dim() != 2 || q.kind() != Kind::Int8 {
            return Err(err("ValueError", "packed weight must be a 2D int8 tensor"));
        }
        if rel.dim() != 2 {
            return Err(err("ValueError", "s_rel must be 2D [N, K/group_size]"));
        }
        if group < 4 || (16 % group != 0 && group % 16 != 0) {
            return Err(err(
                "ValueError",
                format!("group_size={group} must be >=4 and divide 16 or be a multiple of 16"),
            ));
        }
        let geometry = packing::geometry(&q, &rel, group)?;
        let (n, width, bits) = geometry;
        if bits == 6 && (codebook.is_some() || correction.is_some()) {
            return Err(err(
                "ValueError",
                "6-bit storage is uniform and symmetric: no codebook or correction",
            ));
        }
        if width % 16 == 0 && rotation == 0 {
            return Err(err("ZeroDivisionError", "integer modulo by zero"));
        }
        if width % 16 != 0 || width % rotation != 0 {
            return Err(err(
                "ValueError",
                format!("K={width} must be divisible by 16 and convrot_groupsize={rotation}"),
            ));
        }
        for (name, value, expected) in [
            ("s_rel", Some(&rel), vec![n, width / group]),
            ("s_channel", Some(&channel), vec![n]),
            ("correction", correction.as_ref(), vec![width / group, n]),
            ("codebook", codebook.as_ref(), vec![16]),
        ] {
            if let Some(t) = value
                && t.size() != expected
            {
                return Err(err(
                    "ValueError",
                    format!(
                        "{name} must have shape {}, got {}",
                        tuple(&expected),
                        tuple(&t.size())
                    ),
                ));
            }
        }
        Ok(Self {
            q,
            rel,
            channel,
            codebook,
            correction,
            group,
            rotation,
            geometry,
        })
    }
    fn decode(&self, rt: &Runtime, cuda: bool) -> Result<Tensor, Error> {
        let (n, width, bits) = self.geometry;
        if cuda {
            let q = self.q.f_contiguous()?;
            let rel = self.rel.f_contiguous()?;
            let out = Tensor::f_empty([n, width], (Kind::Int8, q.device()))?;
            let cb = self
                .codebook
                .as_ref()
                .map(Tensor::f_contiguous)
                .transpose()?;
            let stream = stream(rt, &q)?;
            let fp8 = rel.kind() == Kind::Float8e4m3fn;
            let rel = if fp8 {
                rel.f_view_dtype(Kind::Uint8)?
            } else {
                rel
            };
            use KernelArg::{Int, Tensor as T};
            unsafe {
                rt.native.call_unchecked(
                    if fp8 {
                        "dequant_int4_grouped_to_int8_e4m3"
                    } else {
                        "dequant_int4_grouped_to_int8"
                    },
                    &[
                        T(&q),
                        T(&rel),
                        cb.as_ref().map_or(KernelArg::None, T),
                        T(&out),
                        Int(self.group),
                        Int(stream),
                    ],
                )?;
            }
            return Ok(out);
        }
        let codes = packing::unpack_codes(&self.q, width, bits)?;
        let values = if let Some(cb) = &self.codebook {
            cb.f_to_device_(self.q.device(), Kind::Float, false, false)?
                .f_index(&[Some(&codes)])?
        } else {
            codes
                .f_to_kind(Kind::Float)?
                .f_sub_scalar((1_i64 << (bits - 1)) as f64)?
        };
        let values = values
            .f_view([n, width / self.group, self.group])?
            .f_mul(&self.rel.f_to_kind(Kind::Float)?.f_unsqueeze(-1)?)?;
        Ok(values
            .f_view([n, width])?
            .f_round()?
            .f_clamp_(-127., 127.)?
            .f_to_kind(Kind::Int8)?)
    }
    fn floating(&self, rt: &Runtime, cuda: bool, kind: Kind) -> Result<Tensor, Error> {
        let (n, width, _) = self.geometry;
        let decoded = self.decode(rt, cuda)?;
        let mut rotated = decoded
            .f_to_kind(Kind::Float)?
            .f_view([n, width / self.group, self.group])?
            .f_mul(&self.channel.f_to_kind(Kind::Float)?.f_view([n, 1, 1])?)?;
        if let Some(c) = &self.correction {
            rotated = rotated.f_add(&c.f_tr()?.f_unsqueeze(-1)?.f_to_kind(Kind::Float)?)?;
        }
        let rotated = rotated.f_view([n, width])?.f_to_kind(kind)?;
        Ok(if cuda {
            unsafe { convrot::rotate_weight(&rt.native, &rotated.f_contiguous()?, self.rotation)? }
        } else {
            int8_eager_ops::rotate_weight(&rotated, self.rotation)?
        }
        .f_to_kind(kind)?)
    }
}
pub(crate) fn run(
    rt: &Runtime,
    cuda: bool,
    linear: bool,
    a: &[Value],
    k: &Kwargs,
) -> Result<Value, Error> {
    if linear {
        let residual = get(a, k, 13, "residual").unwrap_or(&Value::None);
        if !matches!(residual, Value::None) {
            let scale = get(a, k, 14, "residual_scale").unwrap_or(&Value::None);
            let inner_args = a.iter().take(13).cloned().collect::<Vec<_>>();
            let inner_kwargs = k
                .iter()
                .filter(|(name, _)| name != "residual" && name != "residual_scale")
                .cloned()
                .collect();
            let out = run(rt, cuda, true, &inner_args, &inner_kwargs)?;
            return crate::backend_triton_int8::apply_residual(rt, out, residual, scale);
        }
    }
    let x = if linear {
        let input = get(a, k, 0, "x").unwrap();
        Some(
            crate::backend_triton_int8::apply_input(
                rt,
                input,
                get(a, k, 10, "input_act").unwrap_or(&Value::None),
                get(a, k, 11, "input_act_weight").unwrap_or(&Value::None),
                get(a, k, 12, "input_act_eps").unwrap_or(&Value::Float(0.)),
            )?
            .as_tensor()?,
        )
    } else {
        None
    };
    let w = Weight::parse(a, k, linear)?;
    let output = get(
        a,
        k,
        if linear { 9 } else { 7 },
        if linear { "out_dtype" } else { "output_dtype" },
    );
    let kind = if matches!(output, Some(Value::None)) {
        if linear && w.correction.is_some() {
            x.as_ref().unwrap().kind()
        } else {
            Kind::Float
        }
    } else {
        dtype(
            get(
                a,
                k,
                if linear { 9 } else { 7 },
                if linear { "out_dtype" } else { "output_dtype" },
            ),
            Kind::BFloat16,
        )?
    };
    if let Some(x) = x {
        let width = *x
            .size()
            .last()
            .ok_or_else(|| err("IndexError", "tuple index out of range"))?;
        if width != w.geometry.1 {
            return Err(err(
                "ValueError",
                format!("Input K={width} does not match qdata K={}", w.geometry.1),
            ));
        }
        let b = optional_tensor(a, k, 6, "bias")?;
        if cuda {
            if matches!(output, Some(Value::None)) {
                return Err(err("KeyError", "None"));
            }
            return cuda_linear(rt, &w, &x, b.as_ref(), kind, false);
        }
        if w.correction.is_some() {
            let decoded = w.floating(rt, false, x.kind())?;
            return Ok(Value::tensor(
                x.f_linear(&decoded, b.as_ref())?.f_to_kind(kind)?,
            ));
        }
        let decoded = w.decode(rt, false)?;
        let args = [
            Value::tensor(x),
            Value::tensor(decoded),
            Value::tensor(w.channel),
            b.map_or(Value::None, Value::tensor),
            Value::DType(kind),
            Value::Bool(true),
            Value::Int(w.rotation),
        ];
        return super::run(rt, false, "int8_linear", &args, &vec![]);
    }
    Ok(Value::tensor(w.floating(rt, cuda, kind)?))
}
pub(super) fn cuda_linear(
    rt: &Runtime,
    w: &Weight,
    x: &Tensor,
    original_bias: Option<&Tensor>,
    kind: Kind,
    packed: bool,
) -> Result<Value, Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let (n, width, _) = w.geometry;
    let x2 = x.f_reshape([-1, width])?.f_contiguous()?;
    let m = x2.size()[0];
    let code = quantization::dtype_code(kind)?;
    let stream = stream(rt, x)?;
    let mut xq = Tensor::f_empty([m, width], (Kind::Int8, x.device()))?;
    let mut xs = Tensor::f_empty([m, 1], (Kind::Float, x.device()))?;
    let mut out = Tensor::f_empty([m, n], (kind, x.device()))?;
    let bias = original_bias
        .map(|b| crate::cuda_linear::vector(&rt.native, b, x.device(), kind))
        .transpose()?;
    let cb = w.codebook.as_ref().map_or(KernelArg::None, T);
    let mut result_shape = x.size();
    *result_shape.last_mut().unwrap() = n;
    let fp8 = w.rel.kind() == Kind::Float8e4m3fn;
    if !packed
        && rt.w4a8_chunked
        && m <= 8
        && w.correction.is_none()
        && fp8
        && w.group >= 16
        && w.group % 16 == 0
    {
        let rel = w.rel.f_view_dtype(Kind::Uint8)?;
        let used = unsafe {
            rt.native.call_unchecked(
                "w4a8_codebook_gemv",
                &[
                    T(&x2),
                    T(&xq),
                    T(&xs),
                    T(&w.q),
                    T(&rel),
                    cb,
                    T(&w.channel),
                    bias.as_ref().map_or(KernelArg::None, T),
                    T(&out),
                    Int(w.rotation),
                    Int(w.group),
                    Int(code),
                    Int(stream),
                ],
            )?
        }
        .value
            != 0;
        if used {
            return Ok(Value::tensor(out.f_reshape(result_shape.as_slice())?));
        }
    }
    let chunked = packed || rt.w4a8_chunked && w.correction.is_none() && fp8;
    if chunked {
        let device = if let tch::Device::Cuda(i) = x.device() {
            i
        } else {
            return Err(Error::BindingGap {
                backend: "cuda".into(),
                operation: "w4a8 CPU direct call".into(),
            });
        };
        let l2 = rt.native.device_properties(device)?.l2_cache_size;
        let l2 = if l2 == 0 { 1 << 30 } else { l2 };
        let chunk = if l2 < 16 << 20
            && std::env::var_os("COMFY_KITCHEN_INT4_INT8_WEIGHT_CHUNK_N").is_none()
            || n <= 2560
        {
            n
        } else {
            n.min(rt.int4_weight_chunk_n.ok_or_else(|| Error::BindingGap {
                backend: "cuda initialization".into(),
                operation: "dynamic chunk-N integer parsing".into(),
            })?)
        };
        let workspace = Tensor::f_empty([chunk.min(n), width], (Kind::Int8, x.device()))?;
        let fast_act = m >= 512
            && w.rotation == 256
            && width % 256 == 0
            && (256..=16384).contains(&width)
            && convrot::fused_shared_memory_bytes(m, width)
                < convrot::max_dynamic_shared_memory(&rt.native, device)?;
        let rel = w.rel.f_view_dtype(Kind::Uint8)?;
        let used = if fast_act {
            (xq, xs) = unsafe {
                convrot::rowwise64(
                    &rt.native,
                    &x2,
                    w.rotation,
                    Some(0),
                    convrot::InputActivation::None,
                )?
            };
            unsafe {
                rt.native.call_unchecked(
                    "w4a8_codebook_gemm_chunked",
                    &[
                        T(&xq),
                        T(&w.q),
                        T(&rel),
                        w.codebook.as_ref().map_or(KernelArg::None, T),
                        T(&w.channel),
                        T(&xs.f_reshape([m])?),
                        bias.as_ref().map_or(KernelArg::None, T),
                        T(&workspace),
                        T(&out),
                        Int(w.group),
                        Int(chunk),
                        Int(code),
                        Int(stream),
                        Bool(packed), // original nb::arg("mma_packed") default
                    ],
                )?
            }
            .value
                != 0
        } else {
            unsafe {
                rt.native.call_unchecked(
                    "w4a8_codebook_linear_chunked",
                    &[
                        T(&x2),
                        T(&xq),
                        T(&xs),
                        T(&w.q),
                        T(&rel),
                        w.codebook.as_ref().map_or(KernelArg::None, T),
                        T(&w.channel),
                        bias.as_ref().map_or(KernelArg::None, T),
                        T(&workspace),
                        T(&out),
                        Int(w.rotation),
                        Int(w.group),
                        Int(chunk),
                        Int(code),
                        Int(stream),
                        Bool(packed), // original nb::arg("mma_packed") default
                    ],
                )?
            }
            .value
                != 0
        };
        if used {
            return Ok(Value::tensor(out.f_reshape(result_shape.as_slice())?));
        }
        if packed {
            return Err(err(
                "RuntimeError",
                "Packed W4A8 prefill is unsupported by the INT8 GEMM",
            ));
        }
    } else {
        unsafe {
            rt.native.call_unchecked(
                "quantize_int8_rowwise_convrot",
                &[
                    T(&x2),
                    T(&xq),
                    T(&xs),
                    Int(w.rotation),
                    Bool(false),
                    Int(0),
                    Int(stream),
                ],
            )?;
        }
    }
    let decoded = w.decode(rt, true)?;
    let bs = if let Some(b) = bias.as_ref() {
        b.shallow_clone()
    } else {
        crate::cuda_linear::empty(x.device(), kind)?
    };
    let used = unsafe {
        rt.native.call_unchecked(
            "cutlass_int8_dequant",
            &[
                T(&xq),
                T(&decoded),
                T(&xs),
                T(&w.channel),
                T(&bs),
                T(&out),
                Int(code),
                Int(stream),
            ],
        )?
    }
    .value
        != 0;
    if !used {
        let args = [
            Value::tensor(x.shallow_clone()),
            Value::tensor(w.q.shallow_clone()),
            Value::tensor(w.rel.shallow_clone()),
            Value::tensor(w.channel.shallow_clone()),
            w.codebook
                .as_ref()
                .map_or(Value::None, |t| Value::tensor(t.shallow_clone())),
            w.correction
                .as_ref()
                .map_or(Value::None, |t| Value::tensor(t.shallow_clone())),
            original_bias.map_or(Value::None, |t| Value::tensor(t.shallow_clone())),
            Value::Int(w.group),
            Value::Int(w.rotation),
            Value::DType(kind),
        ];
        return run(rt, false, true, &args, &vec![]);
    }
    if let Some(c) = &w.correction {
        let sx = xq
            .f_view([m, width / w.group, w.group])?
            .f_sum_dim_intlist([-1].as_slice(), false, Kind::Int)?
            .f_to_kind(kind)?
            .f_mul(&xs.f_to_kind(kind)?)?;
        let _ = out.f_addmm_(&sx, &c.f_to_kind(kind)?)?;
    }
    Ok(Value::tensor(out.f_reshape(result_shape.as_slice())?))
}
pub(super) fn install(rt: &Runtime) {
    for backend in ["eager", "cuda"] {
        rt.backends.bind_reference(
            backend,
            "dequantize_w4a8_int8_weight",
            body(move |rt, a, k| run(rt, backend == "cuda", false, a, k)),
        );
    }
    for backend in ["eager", "cuda"] {
        rt.backends.bind_reference(
            backend,
            "w4a8_int8_linear",
            body(move |rt, a, k| run(rt, backend == "cuda", true, a, k)),
        );
    }
}
