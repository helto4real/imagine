//! Original CUDA NVFP4 cuBLAS wrapper, including initial-device rejections.
use super::*;
fn run(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let x = tensor(a, kw, 0, "a")?;
    let w = tensor(a, kw, 1, "b")?;
    let sa = tensor(a, kw, 2, "tensor_scale_a")?;
    let sb = tensor(a, kw, 3, "tensor_scale_b")?;
    let ba = tensor(a, kw, 4, "block_scale_a")?;
    let bb = tensor(a, kw, 5, "block_scale_b")?;
    let bias = optional_tensor(a, kw, 6, "bias")?;
    let outkind = get(a, kw, 7, "out_dtype");
    let mut alpha = if let Some(t) = optional_tensor(a, kw, 8, "alpha")? {
        t
    } else {
        sa.f_mul(&sb)?
    };
    if alpha.kind() != Kind::Float {
        alpha = alpha.f_to_kind(Kind::Float)?;
    }
    if alpha.dim() == 0 {
        alpha = alpha.f_reshape([1])?;
    }
    for t in [&x, &w] {
        if t.dim() != 2 {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: if t.dim() > 2 {
                    "too many values to unpack (expected 2)".into()
                } else {
                    format!("not enough values to unpack (expected 2, got {})", t.dim())
                },
            });
        }
    }
    let (m, ka) = (x.size()[0], x.size()[1]);
    let (n, kb) = (w.size()[0], w.size()[1]);
    assertion(ka == kb, "Matrix dimensions do not match")?;
    let kind = if matches!(outkind, None | Some(Value::None)) {
        Kind::Float
    } else {
        dtype(outkind, Kind::Float)?
    };
    let out = Tensor::f_empty([m, n], (kind, x.device()))?;
    assertion(n % 8 == 0, "B tensor must have 8 alignment in N dimension")?;
    let ak = rt.native.tensor_type(&ba)?.scalar_type;
    let bk = rt.native.tensor_type(&bb)?.scalar_type;
    assertion(ak == bk, "A and B scale dtype must match")?;
    if ak == 44 {
        assertion(false, "MXFP4 only supports block length 32")?;
    }
    if ba.f_kind()? == Kind::Float8e4m3fn {
        assertion(alpha.numel() == 1, "alpha must be a scalar")?;
    } else {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: format!(
                "Unsupported scale dtype: {}",
                crate::registry::AllowedType::DType(ba.kind()).python_name()
            ),
        });
    }
    let roundup = |n: i64, g: i64| ((n + g - 1) / g) * g;
    let cols = roundup(ka * 2 / 16, 4);
    assertion(
        ba.dim() == 2 && ba.size() == [roundup(m, 128), cols],
        "Invalid A scale shape",
    )?;
    assertion(
        bb.dim() == 2 && bb.size() == [roundup(n, 128), cols],
        "Invalid B scale shape",
    )?;
    if let Some(b) = &bias {
        assertion(
            matches!(b.kind(), Kind::Half | Kind::BFloat16),
            "Only fp16 and bfloat16 bias are supported.",
        )?;
    }
    let bb = bb.f_view_dtype(Kind::Uint8)?;
    let ba = ba.f_view_dtype(Kind::Uint8)?;
    if matches!(outkind, None | Some(Value::None)) {
        return Err(Error::ReferenceException {
            class: "KeyError",
            message: "None".into(),
        });
    }
    let code = quantization::dtype_code(kind)?;
    let stream = stream(rt, &x)?;
    let b = if let Some(b) = bias.filter(|b| b.numel() != 0) {
        b
    } else {
        Tensor::f_empty([0], (Kind::Half, x.device()))?
    };
    let tch::Device::Cuda(device) = x.device() else {
        unreachable!()
    };
    let workspace = crate::cuda_linear::cublas_workspace(&rt.native, device)?;
    use KernelArg::{Bool, Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            "cublas_gemm_blockwise_fp4",
            &[
                T(&w),
                T(&bb),
                T(&x),
                T(&ba),
                T(&out),
                Int(code),
                T(&b),
                T(&workspace),
                Bool(false),
                T(&alpha),
                Int(stream),
            ],
        )?;
    }
    Ok(Value::tensor(out))
}
pub(super) fn install(rt: &Runtime) {
    rt.backends
        .bind_reference("cuda", "scaled_mm_nvfp4", body(run));
}
