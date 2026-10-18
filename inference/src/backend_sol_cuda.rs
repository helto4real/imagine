//! Frozen CUDA Sol-Attn wrapper, pooled threshold and coarse workspace views.
use super::*;
use std::collections::BTreeMap;
const LOG2E: f64 = std::f64::consts::LOG2_E;
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(crate) fn lengths(
    t: i64,
    n: i64,
    dev: tch::Device,
    block: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let tail = t - (n - 1) * 64;
    let mut out = if let Some(b) = block {
        b.f_to_device_(dev, Kind::Float, false, false)?
            .f_clamp(1, 64)?
    } else {
        Tensor::f_full([n], 64., (Kind::Float, dev))?
    };
    if tail < 64 {
        if block.is_some() {
            out = out.copy();
            let last = out.f_get(-1)?.f_clamp_max(tail)?;
            out.f_get(-1)?.f_copy_(&last)?;
        } else {
            let _ = out.f_get(-1)?.f_fill_(tail)?;
        }
    }
    Ok(out)
}
fn means(x: &Tensor, len: Option<&Tensor>, valid: Option<&Tensor>) -> Result<Tensor, Error> {
    let [b, t, h, d] = super::sol::dims(x)?;
    let full = (t / 64) * 64;
    let x = if let Some(v) = valid {
        x.f_mul(&v.f_view([1, -1, 1, 1])?.f_to_kind(x.kind())?)?
    } else {
        x.shallow_clone()
    };
    let mut m = x
        .f_slice(1, 0, full, 1)?
        .f_view([b, -1, 64, h, d])?
        .f_sum_dim_intlist([2].as_slice(), false, Kind::Float)?;
    if len.is_none() {
        m = m.f_div_scalar(64.)?;
    }
    if full != t {
        let mut tail =
            x.f_slice(1, full, t, 1)?
                .f_sum_dim_intlist([1].as_slice(), true, Kind::Float)?;
        if len.is_none() {
            tail = tail.f_div_scalar(t - full)?;
        }
        m = Tensor::f_cat(&[m, tail], 1)?;
    }
    if let Some(len) = len {
        m = m.f_div(&len.f_view([1, -1, 1, 1])?)?;
    }
    Ok(m.f_permute([0, 2, 1, 3])?.f_reshape([b * h, -1, d])?)
}
fn threshold(
    q: &Tensor,
    k: &Tensor,
    ratio: f64,
    scale: f64,
    len: Option<&Tensor>,
    valid: Option<&Tensor>,
    sinks: (i64, i64),
) -> Result<Tensor, Error> {
    let cen = means(q, len, valid)?;
    let kc = means(k, len, valid)?;
    let kc = kc.f_sub(&kc.f_mean_dim([1].as_slice(), true, None)?)?;
    let csc = cen
        .f_abs()?
        .f_amax([-1], true)?
        .f_div_scalar(127.)?
        .f_clamp_min(1e-8)?;
    let c8 = cen.f_div(&csc)?.f_round()?.f_clamp_(-127., 127.)?;
    topk_from_pooled(
        &c8,
        &csc.f_squeeze_dim(-1)?,
        &kc,
        ratio,
        scale * LOG2E,
        sinks,
    )
}
pub(crate) fn topk_from_pooled(
    c8: &Tensor,
    csc: &Tensor,
    kc: &Tensor,
    ratio: f64,
    log2_scale: f64,
    sinks: (i64, i64),
) -> Result<Tensor, Error> {
    let ksc = kc
        .f_abs()?
        .f_amax([-1], true)?
        .f_div_scalar(127.)?
        .f_clamp_min(1e-12)?;
    let k8 = kc.f_div(&ksc)?.f_round()?.f_clamp_(-127., 127.)?;
    let s = c8
        .f_bmm(&k8.f_transpose(1, 2)?)?
        .f_mul(&csc.f_mul_scalar(log2_scale)?.f_unsqueeze(-1)?)?
        .f_mul(&ksc.f_squeeze_dim(-1)?.f_unsqueeze(-2)?)?;
    let n = kc.size()[1];
    let (s0, s1) = sinks;
    if s1 > s0 {
        let _ = s.f_slice(-1, s0, s1, 1)?.f_fill_(f64::NEG_INFINITY)?;
    }
    let budget = n - (s1.min(n) - s0.min(n)).max(0);
    let kk = 0.max((budget - 1).min(1.max((ratio * budget as f64).round_ties_even() as i64)));
    if kk == 0 {
        return Ok(Tensor::f_full(
            &s.size()[..s.dim() - 1],
            f64::INFINITY,
            (s.kind(), s.device()),
        )?);
    }
    let kth = s.f_topk(kk, -1, true, false)?.0.f_min_dim(-1, false)?.0;
    Ok(kth
        .f_sub(&kth.f_abs()?.f_mul_scalar(1e-5)?)?
        .f_contiguous()?)
}
fn sink(v: Option<&Value>) -> Result<(i64, i64), Error> {
    match v {
        None | Some(Value::None) => Ok((0, 0)),
        Some(Value::List(v) | Value::Tuple(v)) => {
            if v.len() < 2 {
                return Err(error("IndexError", "list index out of range"));
            }
            Ok((integer(v.first(), 0)?, integer(v.get(1), 0)?))
        }
        _ => Err(type_error("'int' object is not subscriptable")),
    }
}
pub(crate) fn coarse(
    out: &mut Tensor,
    workspace: &Tensor,
    p: &BTreeMap<String, i64>,
    len: &Tensor,
    gate: &Tensor,
    scale: f64,
) -> Result<(), Error> {
    let [b, t, h, d] = super::sol::dims(out)?;
    let (bh, n, npad) = (b * h, p["NTB"], p["NPAD"]);
    let qm = workspace
        .f_narrow(0, p["qmean"], bh * npad * d * 4)?
        .f_view_dtype(Kind::Float)?
        .f_view([bh, npad, d])?
        .f_slice(1, 0, n, 1)?;
    let km = workspace
        .f_narrow(0, p["scratch"], bh * npad * d * 4)?
        .f_view_dtype(Kind::Float)?
        .f_view([bh, npad, d])?
        .f_slice(1, 0, n, 1)?;
    let vm = workspace
        .f_narrow(0, p["vcT"], bh * d * npad * 2)?
        .f_view_dtype(Kind::BFloat16)?
        .f_view([bh, d, npad])?
        .f_slice(2, 0, n, 1)?
        .f_transpose(1, 2)?
        .f_to_kind(Kind::Float)?
        .f_div(&len.f_view([1, -1, 1])?)?;
    let s = qm.f_bmm(&km.f_transpose(1, 2)?)?.f_mul_scalar(scale)?;
    let oc = s
        .f_softmax(-1, None)?
        .f_bmm(&vm)?
        .f_view([b, h, -1, d])?
        .f_permute([0, 2, 1, 3])?;
    let gate = gate.f_contiguous()?;
    let full = t / 64;
    if full != 0 {
        let _ = out
            .f_slice(1, 0, full * 64, 1)?
            .f_view([b, full, 64, h, d])?
            .f_addcmul_(
                &gate
                    .f_slice(1, 0, full * 64, 1)?
                    .f_view([b, full, 64, h, d])?,
                &oc.f_slice(1, 0, full, 1)?.f_unsqueeze(2)?,
            )?;
    }
    if t % 64 != 0 {
        let _ = out.f_slice(1, full * 64, t, 1)?.f_addcmul_(
            &gate.f_slice(1, full * 64, t, 1)?,
            &oc.f_slice(1, full, full + 1, 1)?,
        )?;
    }
    Ok(())
}
fn run(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let q = tensor(a, kw, 0, "q")?;
    let k = tensor(a, kw, 1, "k")?;
    let v = tensor(a, kw, 2, "v")?;
    let [b, t, h, d] = super::sol::dims(&q)?;
    if !matches!(q.kind(), Kind::BFloat16 | Kind::Half) {
        return Err(error(
            "ValueError",
            format!(
                "sol_attn: q/k/v must be bfloat16 or float16, got {}",
                crate::registry::AllowedType::DType(q.kind()).python_name()
            ),
        ));
    }
    let tch::Device::Cuda(device) = q.device() else {
        return Err(Error::BindingGap {
            backend: "cuda".into(),
            operation: "Sol CPU direct call".into(),
        });
    };
    let prop = rt.native.device_properties(device)?;
    if prop.major < 8 {
        return Err(error(
            "RuntimeError",
            format!(
                "sol_attn: requires sm_80+ (cp.async, INT8 MMA); cuda:{device} is sm_{}{}",
                prop.major, prop.minor
            ),
        ));
    }
    let mut metadata = crate::registry::CallArgs::new();
    for (i, name) in [
        (5, "sink_blocks"),
        (6, "sink_q"),
        (8, "topk_ratio"),
        (0, "q"),
        (1, "k"),
        (2, "v"),
        (10, "block_len"),
        (11, "coarse_gate"),
    ] {
        if let Some(v) = get(a, kw, i, name) {
            metadata.insert(name.into(), v.metadata(&rt.native)?);
        }
    }
    if let Err(f) = crate::backend_contract::validate_sol(&metadata) {
        if f.raised.is_some() {
            return Err(Error::Selection(
                crate::registry::SelectionError::CallRuleRaised(f),
            ));
        }
        return Err(error(
            "ValueError",
            format!("sol_attn: {}: {}", f.parameter, f.reason),
        ));
    }
    let scale = match get(a, kw, 4, "scale") {
        None | Some(Value::None) => (d as f64).powf(-0.5),
        x => real(x, 0.)?,
    };
    let block = optional_tensor(a, kw, 10, "block_len")?
        .map(|t| t.f_contiguous())
        .transpose()?;
    let len = lengths(t, (t + 63) / 64, q.device(), block.as_ref())?;
    let valid = if block.is_some() {
        Some(super::sol::valid(t, &len)?)
    } else {
        None
    };
    for (name, x) in [("q", &q), ("k", &k), ("v", &v)] {
        if x.stride()[3] != 1 {
            return Err(error(
                "ValueError",
                format!("sol_attn: {name} must have a contiguous last dim"),
            ));
        }
        let align = x.data_ptr() as usize % 16;
        if align != 0 {
            return Err(error(
                "ValueError",
                format!(
                    "sol_attn: {name} must be 16-byte aligned (storage_offset {} leaves it at +{align}); call .contiguous() on it",
                    rt.native.tensor_storage(x)?.offset
                ),
            ));
        }
        for dim in 0..3 {
            if x.size()[dim] > 1 && x.stride()[dim] % 8 != 0 {
                return Err(error(
                    "ValueError",
                    format!(
                        "sol_attn: {name} stride({dim}) = {} elements is not a multiple of 8, so the 16-byte staging loads would be misaligned; call .contiguous() on it",
                        x.stride()[dim]
                    ),
                ));
            }
        }
    }
    let sb = sink(get(a, kw, 5, "sink_blocks"))?;
    let sq = sink(get(a, kw, 6, "sink_q"))?;
    let ratio = real(get(a, kw, 8, "topk_ratio"), 0.)?;
    let thr = if ratio != 0. {
        Some(threshold(
            &q,
            &k,
            ratio,
            scale,
            if block.is_some() { Some(&len) } else { None },
            valid.as_ref(),
            sb,
        )?)
    } else {
        None
    };
    let kb = optional_tensor(a, kw, 7, "key_bias")?
        .map(|kb| -> Result<Tensor, Error> {
            Ok(super::sol::bias(&kb, b, t, q.device())?
                .f_mul_scalar(LOG2E)?
                .f_expand([b, t], false)?
                .f_contiguous()?)
        })
        .transpose()?;
    let mut out = Tensor::f_empty(q.size(), (q.kind(), q.device()))?;
    let aug = integer(get(a, kw, 12, "token_aug"), 0)?;
    use KernelArg::{Bool, Float, Int, Tensor as T};
    let plan = unsafe {
        rt.native
            .call_unchecked("sol_attn_plan", &[Int(b), Int(t), Int(h), Int(aug)])?
    };
    let p: BTreeMap<String, i64> = serde_json::from_str(
        plan.metadata
            .as_deref()
            .ok_or_else(|| Error::Native("Missing native Sol plan".into()))?,
    )
    .map_err(|e| Error::Native(e.to_string()))?;
    let ws = Tensor::f_empty([p["total"]], (Kind::Uint8, q.device()))?;
    unsafe {
        rt.native.call_unchecked(
            "sol_attn",
            &[
                T(&q),
                T(&k),
                T(&v),
                T(&out),
                T(&ws),
                Int(b),
                Int(t),
                Int(h),
                Int(d),
                Float(real(get(a, kw, 3, "tau"), 1.)?),
                Float(scale),
                Int(sb.0),
                Int(sb.1),
                Int(sq.0),
                Int(sq.1),
                Int(stream(rt, &q)?),
                kb.as_ref().map_or(KernelArg::None, T),
                thr.as_ref().map_or(KernelArg::None, T),
                block.as_ref().map_or(KernelArg::None, T),
                Bool(boolean(get(a, kw, 9, "tail"), true)?),
                Int(aug),
            ],
        )?;
    }
    if let Some(gate) = optional_tensor(a, kw, 11, "coarse_gate")? {
        coarse(&mut out, &ws, &p, &len, &gate, scale)?;
    }
    Ok(Value::tensor(out))
}
pub(super) fn install(rt: &Runtime) {
    rt.backends.bind_reference("cuda", "sol_attn", body(run));
}
