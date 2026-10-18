//! Original eager ConvRot W4A4 linear wrapper; CUDA remains a separate body.
use super::*;
use crate::{int8_utils, packing};
fn value_error(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}
fn run(rt: &Runtime, cuda: bool, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let x = tensor(a, k, 0, "x")?;
    let w = tensor(a, k, 1, "qweight")?;
    let ws = tensor(a, k, 2, "wscales")?;
    let bias = optional_tensor(a, k, 3, "bias")?;
    let group = integer(get(a, k, 4, "convrot_groupsize"), 256)?;
    let qgroup = integer(get(a, k, 5, "quant_group_size"), 64)?;
    let linear = get(a, k, 6, "linear_dtype")
        .cloned()
        .unwrap_or(Value::Text("int4".into()));
    if !matches!(&linear,Value::Text(s)if s=="int4"||s=="int8") {
        crate::python_value::ensure_hashable(&linear)?;
        let repr = crate::python_value::repr_with_native(&linear, &rt.native)?;
        return Err(value_error(format!(
            "ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {repr}"
        )));
    }
    if qgroup != 64 {
        return Err(value_error("int4 MMA kernel requires quant_group_size 64"));
    }
    let shape = x.size();
    let width = *shape.last().ok_or_else(|| Error::ReferenceException {
        class: "IndexError",
        message: "tuple index out of range".into(),
    })?;
    let wk = *w.size().last().ok_or_else(|| Error::ReferenceException {
        class: "IndexError",
        message: "tuple index out of range".into(),
    })? * 2;
    if width != wk {
        return Err(value_error(format!(
            "Input K={width} does not match qweight K={wk}"
        )));
    }
    if group == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "integer modulo by zero".into(),
        });
    }
    if width % group != 0 {
        return Err(value_error(format!(
            "Input K={width} not divisible by convrot_groupsize {group}"
        )));
    }
    let x2 = x.f_reshape([-1, width])?.f_contiguous()?;
    if cuda {
        let out = cuda_run(
            rt,
            &x2,
            &w,
            &ws,
            bias.as_ref(),
            group,
            matches!(&linear,Value::Text(s)if s=="int8"),
        )?;
        let mut result = shape[..shape.len() - 1].to_vec();
        result.push(w.size()[0]);
        return Ok(Value::tensor(
            out.f_slice(0, 0, x2.size()[0], 1)?
                .f_reshape(result.as_slice())?,
        ));
    }
    let h = int8_utils::eager_w4_hadamard(group, x2.device(), x2.kind())?;
    let rotated = int8_utils::rotate_activation(&x2, &h, group)?.f_contiguous()?;
    let scale = rotated
        .f_abs()?
        .f_amax([-1], true)?
        .f_clamp_min(1e-10)?
        .f_div_scalar(7.)?;
    let mut codes = rotated.f_div(&scale)?;
    let _ = codes.f_round_()?;
    let codes = codes.f_clamp_(-7., 7.)?.f_to_kind(Kind::Int8)?;
    let packed = packing::pack_int4_row_major(&codes)?;
    let xs = scale.f_reshape([x2.size()[0]])?.f_to_kind(Kind::Float)?;
    let xi = packing::unpack_int4_row_major(&packed, true)?.f_to_kind(x.kind())?;
    let wi = packing::unpack_int4_row_major(&w, true)?.f_to_kind(x.kind())?;
    let out = xi.f_matmul(&wi.f_tr()?)?;
    let out =
        out.f_mul(
            &xs.f_reshape([-1, 1])?
                .f_to_device_(out.device(), x.kind(), false, false)?,
        )?;
    let mut out =
        out.f_mul(
            &ws.f_reshape([1, -1])?
                .f_to_device_(out.device(), x.kind(), false, false)?,
        )?;
    if let Some(b) = bias {
        out = out.f_add(
            &b.f_to_device_(out.device(), x.kind(), false, false)?
                .f_reshape([1, -1])?,
        )?;
    }
    let mut result = shape[..shape.len() - 1].to_vec();
    result.push(w.size()[0]);
    Ok(Value::tensor(
        out.f_slice(0, 0, x2.size()[0], 1)?
            .f_reshape(result.as_slice())?,
    ))
}
pub(super) fn install(rt: &Runtime) {
    for b in ["eager", "cuda"] {
        rt.backends.bind_reference(
            b,
            "convrot_w4a4_linear",
            body(move |rt, a, k| run(rt, b == "cuda", a, k)),
        );
    }
}

fn cuda_run(
    rt: &Runtime,
    x: &Tensor,
    w: &Tensor,
    ws: &Tensor,
    bias: Option<&Tensor>,
    group: i64,
    int8: bool,
) -> Result<Tensor, Error> {
    use crate::{convrot, convrot_w4a4, int4_linear, quantization};
    use KernelArg::{Bool, Int, Tensor as T};
    let tch::Device::Cuda(device) = x.device() else {
        return Err(Error::BindingGap {
            backend: "cuda".into(),
            operation: "ConvRot CPU direct call".into(),
        });
    };
    let props = rt.native.device_properties(device)?;
    // The first-platform native build and existing typed INT4 implementation target SM89.
    if props.major != 8 {
        return Err(Error::BindingGap {
            backend: "cuda".into(),
            operation: "ConvRot INT4 non-SM8x device".into(),
        });
    }
    let width = x.size()[1];
    let m = x.size()[0];
    let n = w.size()[0];
    if !int8 && !rt.force_int4_int8_fallback {
        let (q, xs) = if [16, 64, 256].contains(&group)
            && convrot_w4a4::fused_fits(&rt.native, x, group)?
        {
            quantization::int4_rowwise(
                &rt.native,
                x,
                quantization::Int4Rows::ConvRot(group),
                Some(0),
            )?
        } else {
            let h = int8_utils::hadamard(group, x.device(), x.kind())?;
            let rot = int8_utils::rotate_activation(x, &h, group)?.f_contiguous()?;
            quantization::int4_rowwise(&rt.native, &rot, quantization::Int4Rows::Plain, Some(0))?
        };
        return unsafe {
            int4_linear::linear(
                &rt.native,
                &int4_linear::Linear {
                    activation: &q,
                    weight: w,
                    activation_scale: &xs,
                    weight_scale: ws,
                    bias,
                    output_kind: x.kind(),
                },
                &int4_linear::Settings {
                    force_int8_fallback: rt.force_int4_int8_fallback,
                    disable_cutlass: rt.disable_cutlass_int8,
                    weight_chunk_n: rt.int4_weight_chunk_n.unwrap_or(4096),
                },
            )
        };
    }
    let (q, xs) = if group == 256
        && width % 256 == 0
        && (256..=16384).contains(&width)
        && convrot::fused_shared_memory_bytes(m, width)
            < convrot::max_dynamic_shared_memory(&rt.native, device)?
    {
        unsafe {
            convrot::rowwise64(
                &rt.native,
                x,
                group,
                Some(0),
                convrot::InputActivation::None,
            )?
        }
    } else {
        let h = int8_utils::hadamard(group, x.device(), x.kind())?;
        convrot::quantize_activation(&rt.native, x, &h, group, Some(0))?
    };
    let small = m <= 8;
    let label = if small {
        "packed INT4 weight GEMV"
    } else {
        "chunked INT4 weight GEMM"
    };
    if w.dim() != 2 || q.size()[1] != w.size()[1] * 2 {
        return Err(value_error(format!("{label} K dimensions do not match")));
    }
    if !small && (!q.is_contiguous() || !w.is_contiguous()) {
        return Err(value_error(
            "chunked INT4 weight GEMM expects contiguous activation and weight tensors",
        ));
    }
    let chunk = if n <= 2560 {
        n
    } else {
        n.min(rt.int4_weight_chunk_n.ok_or_else(|| Error::BindingGap {
            backend: "cuda initialization".into(),
            operation: "dynamic chunk-N integer parsing".into(),
        })?)
    };
    let out = Tensor::f_empty([m, n], (x.kind(), x.device()))?;
    let workspace = if small {
        None
    } else {
        Some((
            Tensor::f_empty([chunk, width], (Kind::Int8, x.device()))?,
            Tensor::f_empty([m, chunk], (Kind::Int, x.device()))?,
        ))
    };
    let xs = xs
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_reshape([-1, 1])?
        .f_contiguous()?;
    if xs.size()[0] != m {
        return Err(value_error(format!(
            "{label} x_scale must have {m} values, got {}",
            xs.size()[0]
        )));
    }
    let ws = ws
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_reshape([-1])?
        .f_contiguous()?;
    if ws.numel() != n as usize {
        return Err(value_error(format!(
            "{label} weight_scale must have {n} values, got {}",
            ws.numel()
        )));
    }
    let b = if let Some(b) = bias {
        if small {
            if b.device() != x.device() || b.kind() != x.kind() || !b.is_contiguous() {
                b.f_to_device_(x.device(), x.kind(), false, false)?
                    .f_contiguous()?
            } else {
                b.shallow_clone()
            }
        } else {
            crate::cuda_linear::vector(&rt.native, b, x.device(), x.kind())?
        }
    } else {
        crate::cuda_linear::empty(x.device(), x.kind())?
    };
    let code = quantization::dtype_code(x.kind())?;
    let stream = stream(rt, x)?;
    if let Some((ww, acc)) = workspace {
        let cublas = crate::cuda_linear::cublas_workspace(&rt.native, device)?;
        unsafe {
            rt.native.call_unchecked(
                "int4_weight_int8_act_gemm_dequant_chunked",
                &[
                    T(&q),
                    T(w),
                    T(&xs),
                    T(&ws),
                    T(&b),
                    T(&out),
                    T(&ww),
                    T(&acc),
                    T(&cublas),
                    Int(chunk),
                    Bool(true),
                    Int(code),
                    Int(stream),
                ],
            )?;
        }
    } else {
        unsafe {
            rt.native.call_unchecked(
                "int4_weight_int8_act_gemv_dequant",
                &[
                    T(&q),
                    T(&w.f_contiguous()?),
                    T(&xs),
                    T(&ws),
                    T(&b),
                    T(&out),
                    Int(code),
                    Int(stream),
                ],
            )?;
        }
    }
    Ok(out)
}
