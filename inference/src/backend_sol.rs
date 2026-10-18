//! Frozen eager Sol-Attn, including the original atomic scatter-add.
use super::*;
const BLOCK: i64 = 64;
// Identical FP64 bits to the frozen Python literal 1.4426950408889634.
const LOG2E: f64 = std::f64::consts::LOG2_E;
const NEG: f64 = f32::MIN as f64;
fn err(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(super) fn dims(t: &Tensor) -> Result<[i64; 4], Error> {
    t.size().try_into().map_err(|_| {
        err(
            "ValueError",
            if t.dim() < 4 {
                format!("not enough values to unpack (expected 4, got {})", t.dim())
            } else {
                "too many values to unpack (expected 4)".into()
            },
        )
    })
}
fn sinks(v: Option<&Value>) -> Result<(i64, i64), Error> {
    match v {
        None | Some(Value::None) => Ok((0, 0)),
        Some(Value::List(v) | Value::Tuple(v)) if v.is_empty() => Ok((0, 0)),
        Some(Value::List(v) | Value::Tuple(v)) => {
            if v.len() < 2 {
                return Err(err(
                    "ValueError",
                    format!("not enough values to unpack (expected 2, got {})", v.len()),
                ));
            }
            Ok((integer(v.first(), 0)?, integer(v.get(1), 0)?))
        }
        _ => Err(type_error("'int' object is not subscriptable")),
    }
}
pub(super) fn valid(t: i64, lengths: &Tensor) -> Result<Tensor, Error> {
    let pos = Tensor::f_arange(t, (Kind::Int64, lengths.device()))?;
    Ok(pos
        .f_remainder(BLOCK)?
        .f_lt_tensor(&lengths.f_index_select(0, &pos.f_floor_divide_scalar(BLOCK)?)?)?)
}
fn pool(x: &Tensor, n: i64, lengths: &Tensor, sum: bool) -> Result<Tensor, Error> {
    let [b, t, h, d] = dims(x)?;
    let mut x = x.f_mul(&valid(t, lengths)?.f_view([1, -1, 1, 1])?)?;
    let pad = n * BLOCK - t;
    if pad != 0 {
        x = Tensor::f_cat(
            &[&x, &x.f_new_zeros([b, pad, h, d], (x.kind(), x.device()))?],
            1,
        )?;
    }
    let out = x
        .f_reshape([b, n, BLOCK, h, d])?
        .f_sum_dim_intlist([2].as_slice(), false, None)?;
    Ok(if sum {
        out
    } else {
        out.f_div(&lengths.f_view([1, -1, 1, 1])?)?
    })
}
fn slice(t: &Tensor, dim: i64, start: i64, end: i64) -> Result<Tensor, Error> {
    Ok(t.f_slice(dim, start, end, 1)?)
}
pub(super) fn bias(kb: &Tensor, b: i64, t: i64, dev: tch::Device) -> Result<Tensor, Error> {
    let devname = |d| match d {
        tch::Device::Cpu => "cpu".into(),
        tch::Device::Cuda(i) => format!("cuda:{i}"),
        _ => format!("{d:?}"),
    };
    let tuple = |s: Vec<i64>| {
        format!(
            "({}{})",
            s.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
            if s.len() == 1 { "," } else { "" }
        )
    };
    if kb.device() != dev {
        return Err(err(
            "ValueError",
            format!(
                "sol_attn: key_bias must be on {}, got {}",
                devname(dev),
                devname(kb.device())
            ),
        ));
    }
    let mut kb = kb.shallow_clone();
    if kb.dim() == 4 {
        if kb.size()[1] != 1 || kb.size()[2] != 1 {
            return Err(err(
                "ValueError",
                format!(
                    "sol_attn: key_bias must be key-only; a mask varying over heads or queries ({}) cannot be expressed by this op -- use a dense attention for those calls",
                    tuple(kb.size())
                ),
            ));
        }
        kb = kb.f_select(1, 0)?.f_select(1, 0)?;
    }
    if kb.dim() == 1 {
        kb = kb.f_unsqueeze(0)?;
    }
    if kb.dim() != 2 || kb.size()[1] != t || ![1, b].contains(&kb.size()[0]) {
        return Err(err(
            "ValueError",
            format!(
                "sol_attn: key_bias must be (T,), (B, T) or (B, 1, 1, T), got {} for T={t}, B={b}",
                tuple(kb.size())
            ),
        ));
    }
    if kb.kind() == Kind::Bool {
        let zeros = Tensor::f_zeros(kb.size(), (Kind::Float, dev))?;
        kb = zeros.f_where_self(&kb, &zeros.f_full_like(f64::NEG_INFINITY)?)?;
    }
    Ok(kb.f_to_kind(Kind::Float)?)
}
fn run(rt: &Runtime, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let q = tensor(a, k, 0, "q")?;
    let key = tensor(a, k, 1, "k")?;
    let v = tensor(a, k, 2, "v")?;
    let [b, t, h, d] = dims(&q)?;
    let n = (t + BLOCK - 1) / BLOCK;
    let tau = real(get(a, k, 3, "tau"), 1.)?;
    let scale = match get(a, k, 4, "scale") {
        None | Some(Value::None) => (d as f64).powf(-0.5),
        x => real(x, 0.)?,
    };
    let log2s = scale * LOG2E;
    let (sk0, sk1) = sinks(get(a, k, 5, "sink_blocks"))?;
    let (sq0, sq1) = sinks(get(a, k, 6, "sink_q"))?;
    let score_bytes = b as i128 * h as i128 * t as i128 * t as i128 * 4;
    if score_bytes > 4 * 2_i128.pow(30) {
        let dtype = match q.kind() {
            Kind::Float => "torch.float32",
            Kind::Half => "torch.float16",
            Kind::BFloat16 => "torch.bfloat16",
            Kind::Double => "torch.float64",
            _ => "unknown",
        };
        let device = if q.device() == tch::Device::Cpu {
            "cpu"
        } else {
            "cuda"
        };
        return Err(err(
            "RuntimeError",
            format!(
                "sol_attn: the eager reference is O(T^2) and would need {:.1} GiB for the score tensor at (B={b}, H={h}, T={t}). It was selected because no fused backend accepted these inputs -- the fused backends take bfloat16 or float16 with head_dim 128, and got {dtype} on {device}.",
                score_bytes as f64 / 2_f64.powi(30)
            ),
        ));
    }
    let fq = q.f_to_kind(Kind::Float)?;
    let fk = key.f_to_kind(Kind::Float)?;
    let fv = v.f_to_kind(Kind::Float)?;
    let block_len = optional_tensor(a, k, 10, "block_len")?;
    let tail_len = t - (n - 1) * BLOCK;
    let mut lengths = match &block_len {
        Some(x) => x
            .f_to_device_(q.device(), Kind::Float, false, false)?
            .f_clamp(1, 64)?,
        None => Tensor::f_full([n], 64., (Kind::Float, q.device()))?,
    };
    if tail_len < BLOCK {
        if block_len.is_some() {
            lengths = lengths.copy();
            let last = lengths.f_select(0, -1)?.f_clamp_max(tail_len)?;
            lengths.f_select(0, -1)?.f_copy_(&last)?;
        } else {
            let _ = lengths.f_select(0, -1)?.f_fill_(tail_len)?;
        }
    }
    let kc = pool(&fk, n, &lengths, false)?;
    let vc = pool(&fv, n, &lengths, true)?;
    let km = kc.f_mean_dim([1].as_slice(), true, None)?;
    let kcc = kc.f_sub(&km)?;
    let kvar = kcc
        .f_pow_tensor_scalar(2)?
        .f_mean_dim([1].as_slice(), false, None)?;
    let centroid = pool(&fq, n, &lengths, false)?;
    let qh = fq.f_permute([0, 2, 1, 3])?;
    let kh = fk.f_sub(&km)?.f_permute([0, 2, 1, 3])?;
    let vh = fv.f_permute([0, 2, 1, 3])?;
    let kch = kcc.f_permute([0, 2, 1, 3])?;
    let vch = vc.f_permute([0, 2, 1, 3])?;
    let mut stok = qh.f_matmul(&kh.f_transpose(-1, -2)?)?.f_mul_scalar(log2s)?;
    if block_len.is_some() {
        stok = stok.f_masked_fill(
            &valid(t, &lengths)?.f_view([1, 1, 1, t])?.f_logical_not()?,
            NEG,
        )?;
    }
    if let Some(kb) = optional_tensor(a, k, 7, "key_bias")? {
        stok = stok.f_add(
            &bias(&kb, b, t, q.device())?
                .f_mul_scalar(LOG2E)?
                .f_reshape([-1, 1, 1, t])?,
        )?;
    }
    let mut sblk = qh
        .f_matmul(&kch.f_transpose(-1, -2)?)?
        .f_mul_scalar(log2s)?;
    let qb = Tensor::f_arange(t, (Kind::Int64, q.device()))?.f_floor_divide_scalar(BLOCK)?;
    if block_len.is_some() {
        sblk = sblk.f_mul(&valid(t, &lengths)?.f_view([1, 1, t, 1])?)?;
    }
    let index = qb.f_view([1, 1, t, 1])?.f_expand([b, h, t, n], false)?;
    let mut col = Tensor::f_zeros([b, h, n, n], (sblk.kind(), q.device()))?;
    rt.sol_scatter(&mut col, &index, &sblk)?;
    let col = col.f_div(&lengths.f_view([1, 1, n, 1])?)?;
    let idx = Tensor::f_arange(n, (Kind::Int64, q.device()))?;
    let ratio = real(get(a, k, 8, "topk_ratio"), 0.)?;
    let mut exact = if ratio != 0. {
        let ranked = col.copy();
        let _ = slice(&ranked, -1, sk0, sk1)?.f_fill_(f64::NEG_INFINITY)?;
        let count = (sk1.min(n) - sk0.min(n)).max(0);
        let nn = n - count;
        let r = ratio * nn as f64;
        if !r.is_finite() {
            return Err(err(
                if r.is_nan() {
                    "ValueError"
                } else {
                    "OverflowError"
                },
                if r.is_nan() {
                    "cannot convert float NaN to integer"
                } else {
                    "cannot convert float infinity to integer"
                },
            ));
        }
        let kk = (nn - 1).min((r.round_ties_even() as i64).max(1)).max(0);
        if kk != 0 {
            let (values, _) = ranked.f_topk(kk, -1, true, true)?;
            ranked.f_ge_tensor(&slice(&values, -1, -1, i64::MAX)?)?
        } else {
            ranked.f_zeros_like()?.f_to_kind(Kind::Bool)?
        }
    } else {
        let var = centroid
            .f_pow_tensor_scalar(2)?
            .f_mul(&kvar.f_unsqueeze(1)?)?
            .f_sum_dim_intlist([-1].as_slice(), false, None)?;
        let thr = var
            .f_mul_scalar(log2s)?
            .f_mul_scalar(log2s)?
            .f_add_scalar(1e-6)?
            .f_sqrt()?
            .f_mul_scalar(tau)?;
        col.f_gt_tensor(&thr.f_permute([0, 2, 1])?.f_unsqueeze(-1)?)?
    };
    let _ = exact.f_logical_or_(
        &idx.f_view([1, -1])?
            .f_sub(&idx.f_view([-1, 1])?)?
            .f_abs()?
            .f_le(1)?
            .f_view([1, 1, n, n])?,
    )?;
    let _ = exact.f_logical_or_(
        &idx.f_ge(sk0)?
            .f_logical_and(&idx.f_lt(sk1)?)?
            .f_view([1, 1, 1, n])?,
    )?;
    let _ = exact.f_logical_or_(
        &idx.f_ge(sq0)?
            .f_logical_and(&idx.f_lt(sq1)?)?
            .f_view([1, 1, n, 1])?,
    )?;
    let ex = exact.f_gather(2, &index, false)?;
    let keep = slice(&ex.f_repeat_interleave_self_int(BLOCK, -1, None)?, -1, 0, t)?;
    stok = stok.f_masked_fill(&keep.f_logical_not()?, NEG)?;
    sblk = col.f_gather(2, &index, false)?.f_masked_fill(&ex, NEG)?;
    if !boolean(get(a, k, 9, "tail"), true)? {
        sblk = sblk.f_full_like(NEG)?;
    }
    let logits = Tensor::f_cat(&[stok, sblk], -1)?;
    let p = logits
        .f_sub(&logits.f_amax([-1], true)?)?
        .f_exp2()?
        .f_masked_fill(&logits.f_le(NEG)?, 0.)?;
    let pt = slice(&p, -1, 0, t)?;
    let pb = slice(&p, -1, t, i64::MAX)?;
    let num = pt.f_matmul(&vh)?.f_add(&pb.f_matmul(&vch)?)?;
    let den = pt.f_sum_dim_intlist([-1].as_slice(), false, None)?.f_add(
        &pb.f_mul(&lengths.f_view([1, 1, 1, n])?)?
            .f_sum_dim_intlist([-1].as_slice(), false, None)?,
    )?;
    let out = num
        .f_div(&den.f_clamp_min(1e-30)?.f_unsqueeze(-1)?)?
        .f_permute([0, 2, 1, 3])?
        .f_to_kind(v.kind())?
        .f_contiguous()?;
    if let Some(gate) = optional_tensor(a, k, 11, "coarse_gate")? {
        let flat = |x: &Tensor| -> Result<Tensor, Error> {
            Ok(x.f_permute([0, 2, 1, 3])?.f_reshape([b * h, n, d])?)
        };
        let qm = flat(&centroid)?;
        let km = flat(&kc)?;
        let vm = flat(&vc.f_div(&lengths.f_view([1, -1, 1, 1])?)?)?;
        let score = qm.f_bmm(&km.f_transpose(1, 2)?)?.f_mul_scalar(scale)?;
        let oc = score
            .f_softmax(-1, None)?
            .f_bmm(&vm)?
            .f_view([b, h, -1, d])?
            .f_permute([0, 2, 1, 3])?;
        let gate = gate.f_contiguous()?;
        let full = t / BLOCK;
        if full != 0 {
            let _ = slice(&out, 1, 0, full * BLOCK)?
                .f_view([b, full, BLOCK, h, d])?
                .f_addcmul_(
                    &slice(&gate, 1, 0, full * BLOCK)?.f_view([b, full, BLOCK, h, d])?,
                    &slice(&oc, 1, 0, full)?.f_unsqueeze(2)?,
                )?;
        }
        if t % BLOCK != 0 {
            let _ = slice(&out, 1, full * BLOCK, i64::MAX)?.f_addcmul_(
                &slice(&gate, 1, full * BLOCK, i64::MAX)?,
                &slice(&oc, 1, full, full + 1)?,
            )?;
        }
    }
    Ok(Value::tensor(out))
}
pub(super) fn install(rt: &Runtime) {
    rt.backends.bind_reference("eager", "sol_attn", body(run));
}
