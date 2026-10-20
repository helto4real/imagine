//! Frozen eager W4A8/W6A8 preparation and CUDA fused/staged dispatch.
use super::seeds::Seed;
use super::*;
use crate::{convrot, packing};

pub(super) const LUT: [f32; 16] = [
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
];
const BUDGET: i64 = 1 << 22;
fn fail(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn sum(x: &Tensor) -> Result<Tensor, Error> {
    Ok(x.f_sum_dim_intlist([-1].as_slice(), true, None)?)
}
fn rotate(rt: &Runtime, cuda: bool, x: &Tensor, group: i64) -> Result<Tensor, Error> {
    let x = x.f_contiguous()?;
    if cuda {
        unsafe { convrot::rotate_weight(&rt.native, &x, group) }
    } else {
        int8_eager_ops::rotate_weight(&x, group)
    }
}
fn sample(rt: &Runtime, x: &Tensor, size: i64) -> Result<Tensor, Error> {
    let x = x.f_flatten(0, -1)?;
    Ok(if x.numel() > size as usize {
        let idx = rt
            .native
            .generator(x.device(), 0)?
            .randint(x.numel() as i64, &[size])?;
        x.f_index(&[Some(&idx)])?
    } else {
        x
    })
}
fn fit(rt: &Runtime, normalized: &Tensor) -> Result<Tensor, Error> {
    let samples = sample(rt, normalized, 300000)?.f_to_kind(Kind::Float)?;
    let q = Tensor::f_linspace(0., 1., 16, (Kind::Float, samples.device()))?;
    let mut cb = samples.f_quantile(&q, None, false, "linear")?;
    for _ in 0..25 {
        let assignments = samples
            .f_unsqueeze(-1)?
            .f_sub(&cb)?
            .f_abs()?
            .f_argmin(-1, false)?;
        let updated = rt.native.clone_tensor(&cb)?;
        for index in 0..16 {
            let selected = assignments.f_eq(index)?;
            if selected.f_any()?.int64_value(&[]) != 0 {
                updated
                    .f_get(index)?
                    .f_copy_(&samples.f_masked_select(&selected)?.f_mean(None)?)?;
            }
        }
        cb = updated;
    }
    Ok(cb.f_contiguous()?)
}
fn codebook(rt: &Runtime, normalized: &Tensor) -> Result<Tensor, Error> {
    let x = sample(rt, &normalized.detach(), 1 << 19)?.f_to_kind(Kind::Float)?;
    let kurtosis = x
        .f_sub(&x.f_mean(None)?)?
        .f_div(&x.f_std(true)?.f_add_scalar(1e-9)?)?
        .f_pow_tensor_scalar(4)?
        .f_mean(None)?
        .f_sub_scalar(3.)?
        .double_value(&[]);
    if kurtosis <= -0.1 {
        Ok(Tensor::f_from_slice(&LUT)?.f_to_device(normalized.device())?)
    } else {
        fit(rt, normalized)
    }
}
fn decide(
    rt: &Runtime,
    cuda: bool,
    w: &Tensor,
    group: i64,
    rotation: i64,
) -> Result<Tensor, Error> {
    let (n, width) = (w.size()[0], w.size()[1]);
    let max_rows = (BUDGET / width.max(1)).max(1);
    let rows = if n > max_rows {
        let idx = Tensor::f_linspace(0., (n - 1) as f64, max_rows, (Kind::Float, w.device()))?
            .f_to_kind(Kind::Int64)?;
        w.f_index_select(0, &idx)?
    } else {
        w.shallow_clone()
    };
    let grouped = rotate(rt, cuda, &rows, rotation)?
        .f_to_kind(Kind::Float)?
        .f_view([rows.size()[0], width / group, group])?;
    codebook(
        rt,
        &grouped.f_div(&grouped.f_abs()?.f_amax([-1], true)?.f_clamp_min(1e-8)?)?,
    )
}
fn search(values: &Tensor, levels: &Tensor) -> Result<Tensor, Error> {
    Ok(values.f_searchsorted(levels, false, false, "left", None::<&Tensor>)?)
}
fn assign_codes(x: &Tensor, cb: &Tensor) -> Result<Tensor, Error> {
    let last = cb.numel() as i64 - 1;
    let pos = search(&x.f_contiguous()?, cb)?;
    let lo = pos.f_sub_scalar(1)?.f_clamp(0, last)?;
    let hi = pos.f_clamp(0, last)?;
    let dlo = x.f_sub(&cb.f_index(&[Some(&lo)])?)?.f_abs_()?;
    let dhi = x.f_sub(&cb.f_index(&[Some(&hi)])?)?.f_abs_()?;
    Ok(hi
        .f_where_self(&dhi.f_lt_tensor(&dlo)?, &lo)?
        .f_to_kind(Kind::Int)?)
}
fn round(rt: &Runtime, x: &Tensor, seed: &Seed) -> Result<Tensor, Error> {
    if seed.positive(rt)? {
        let noise = rt
            .native
            .generator(x.device(), seed.generator(rt)?)?
            .rand(&x.size(), x.kind())?;
        Ok(x.f_add(&noise)?.f_floor()?)
    } else {
        Ok(x.f_round()?)
    }
}
fn levels(table: &Tensor, rel: &Tensor) -> Result<Tensor, Error> {
    Ok(table
        .f_view([1, 1, -1])?
        .f_mul(&rel.f_to_kind(Kind::Float)?.f_unsqueeze(-1)?)?
        .f_round_()?
        .f_clamp_(-127., 127.)?)
}
fn assign_grid(
    rt: &Runtime,
    w: &Tensor,
    lv: &Tensor,
    channel: &Tensor,
    seed: &Seed,
    target: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let shape = w.size();
    let last = lv.size()[2] - 1;
    let lv = lv
        .f_reshape([shape[0] * shape[1], last + 1])?
        .f_contiguous()?;
    let target = if let Some(t) = target {
        t.shallow_clone()
    } else {
        w.f_div(&channel.f_view([-1, 1, 1])?)?
    };
    let tg = target
        .f_reshape([shape[0] * shape[1], shape[2]])?
        .f_contiguous()?;
    let pos = search(&tg, &lv)?;
    let idx = if seed.positive(rt)? {
        let lo = pos.f_sub_scalar(1)?.f_clamp_(0, last - 1)?;
        let level_lo = lv.f_gather(1, &lo, false)?;
        let width = lv
            .f_gather(1, &lo.f_add_scalar(1)?, false)?
            .f_sub_(&level_lo)?;
        let mut thr = rt
            .native
            .generator(tg.device(), seed.generator(rt)?)?
            .rand(&tg.size(), tg.kind())?;
        let _ = thr.f_mul_(&width)?.f_add_(&level_lo)?;
        lo.f_add(&tg.f_gt_tensor(&thr)?)?.f_clamp_(0, last)?
    } else {
        let lo = pos.f_sub_scalar(1)?.f_clamp(0, last)?;
        let hi = pos.f_clamp(0, last)?;
        let dlo = tg.f_sub(&lv.f_gather(1, &lo, false)?)?.f_abs_()?;
        let dhi = tg.f_sub(&lv.f_gather(1, &hi, false)?)?.f_abs_()?;
        hi.f_where_self(&dhi.f_lt_tensor(&dlo)?, &lo)?
    };
    Ok(idx.f_to_kind(Kind::Int)?.f_reshape(shape.as_slice())?)
}
fn score(
    rt: &Runtime,
    w: &Tensor,
    rel: Tensor,
    ch: &Tensor,
    table: &Tensor,
    target: &Tensor,
) -> Result<(Tensor, Tensor, Tensor), Error> {
    let lv = levels(table, &rel)?;
    let idx = assign_grid(rt, w, &lv, ch, &Seed::zero(), Some(target))?;
    let err = lv
        .f_gather(2, &idx.f_to_kind(Kind::Int64)?, false)?
        .f_sub_(target)?
        .f_pow_(2)?
        .f_sum_dim_intlist([-1].as_slice(), false, None)?;
    Ok((rel, err, idx))
}
fn scale_search(
    rt: &Runtime,
    w: &Tensor,
    rel: &Tensor,
    ch: &Tensor,
    table: &Tensor,
) -> Result<(Tensor, Tensor), Error> {
    let target = w.f_div(&ch.f_view([-1, 1, 1])?)?;
    let candidates = if rel.kind() == Kind::Float8e4m3fn {
        let raw = rel.f_view_dtype(Kind::Uint8)?;
        vec![
            rel.f_to_kind(Kind::Float)?,
            raw.f_sub_scalar(1)?
                .f_clamp_min_(1)?
                .f_view_dtype(Kind::Float8e4m3fn)?
                .f_to_kind(Kind::Float)?,
            raw.f_add_scalar(1)?
                .f_clamp_max_(0x7e)?
                .f_view_dtype(Kind::Float8e4m3fn)?
                .f_to_kind(Kind::Float)?,
        ]
    } else {
        let base = rel.f_to_kind(Kind::Float)?;
        vec![
            base.shallow_clone(),
            base.f_mul_scalar(1. - 0.03)?,
            base.f_mul_scalar(1. + 0.03)?,
        ]
    };
    let mut it = candidates.into_iter();
    let (mut best_scale, mut best_err, mut best_idx) =
        score(rt, w, it.next().unwrap(), ch, table, &target)?;
    for cand in it {
        let (cand, err, idx) = score(rt, w, cand, ch, table, &target)?;
        let better = err.f_lt_tensor(&best_err)?;
        best_err.f_copy_(&err.f_where_self(&better, &best_err)?)?;
        best_scale.f_copy_(&cand.f_where_self(&better, &best_scale)?)?;
        best_idx.f_copy_(&idx.f_where_self(&better.f_unsqueeze(-1)?, &best_idx)?)?;
    }
    Ok((best_scale.f_to_kind(rel.kind())?.f_contiguous()?, best_idx))
}
struct Options {
    group: i64,
    rotation: i64,
    symmetric: bool,
    scale: Kind,
    codebook: bool,
    override_cb: Option<Tensor>,
    seed: Seed,
    bits: i64,
    search: bool,
}
type Prepared = (Tensor, Tensor, Tensor, Option<Tensor>, Option<Tensor>);
fn rotated(
    rt: &Runtime,
    w: &Tensor,
    o: &Options,
    cb: Option<&Tensor>,
    seed: &Seed,
) -> Result<Prepared, Error> {
    if !matches!(o.scale, Kind::Float | Kind::Float8e4m3fn) {
        return Err(fail(
            "ValueError",
            format!(
                "scale_dtype must be float32 or float8_e4m3fn, got {}",
                crate::registry::AllowedType::DType(o.scale).python_name()
            ),
        ));
    }
    let (n, k) = (w.size()[0], w.size()[1]);
    let groups = k / o.group;
    let gw = w.f_to_kind(Kind::Float)?.f_view([n, groups, o.group])?;
    let mut cb_out = None;
    let mut table = None;
    let mut unsigned = None;
    let (group_scale, shifted, correction) = if o.bits == 6 || (o.symmetric && o.codebook) {
        let amax = gw.f_abs()?.f_amax([-1], true)?;
        let mut gs = if o.bits == 6 {
            amax.f_div_scalar(31.)?.f_clamp_min(1e-8)?
        } else {
            amax.f_clamp_min(1e-8)?
        };
        let lv = if o.bits == 6 {
            Tensor::f_arange_start(-31, 32, (Kind::Float, w.device()))?
        } else {
            let normalized = gw.f_div(&gs)?;
            let c = if let Some(c) = cb {
                c.shallow_clone()
            } else {
                codebook(rt, &normalized)?
            };
            cb_out = Some(c.shallow_clone());
            c
        };
        let assign = |x: &Tensor| -> Result<Tensor, Error> {
            if o.bits == 6 {
                Ok(x.f_round()?
                    .f_clamp_(-31., 31.)?
                    .f_add_scalar(31)?
                    .f_to_kind(Kind::Int)?)
            } else {
                assign_codes(x, &lv)
            }
        };
        let mut q = assign(&gw.f_div(&gs)?)?;
        for _ in 0..2 {
            let ql = lv.f_index(&[Some(&q)])?;
            gs = sum(&gw.f_mul(&ql)?)?
                .f_div(&sum(&ql.f_mul(&ql)?)?.f_clamp_min(1e-8)?)?
                .f_clamp_min(1e-8)?;
            q = assign(&gw.f_div(&gs)?)?;
        }
        let shifted = lv.f_index(&[Some(&q)])?.f_mul(&gs)?;
        table = Some(lv);
        (gs, shifted, None)
    } else if o.symmetric {
        let gs = gw
            .f_abs()?
            .f_amax([-1], true)?
            .f_div_scalar(7.)?
            .f_clamp_min(1e-8)?;
        let signed = round(rt, &gw.f_div(&gs)?, seed)?
            .f_clamp(-8., 7.)?
            .f_to_kind(Kind::Int)?;
        unsigned = Some(signed.f_add_scalar(8)?.f_view([n, k])?);
        let shifted = signed.f_mul(&gs)?;
        (gs, shifted, None)
    } else {
        let minimum = gw.f_amin([-1], true)?;
        let gs = gw
            .f_amax([-1], true)?
            .f_sub(&minimum)?
            .f_div_scalar(15.)?
            .f_clamp_min(1e-8)?;
        let u = round(rt, &gw.f_sub(&minimum)?.f_div(&gs)?, seed)?
            .f_clamp(0., 15.)?
            .f_to_kind(Kind::Int)?
            .f_view([n, k])?;
        let shifted = u
            .f_view([n, groups, o.group])?
            .f_sub_scalar(8)?
            .f_mul(&gs)?;
        let corr = gs
            .f_mul_scalar(8.)?
            .f_add(&minimum)?
            .f_squeeze_dim(-1)?
            .f_tr()?
            .f_contiguous()?
            .f_to_kind(w.kind())?;
        unsigned = Some(u);
        (gs, shifted, Some(corr))
    };
    let ch = shifted
        .f_abs()?
        .f_amax([1, 2], false)?
        .f_div_scalar(127.)?
        .f_clamp_min(1e-8)?;
    let mut rel = group_scale
        .f_squeeze_dim(-1)?
        .f_div(&ch.f_unsqueeze(1)?)?
        .f_to_kind(Kind::Float)?
        .f_contiguous()?;
    if o.scale != Kind::Float {
        rel = rel.f_to_kind(o.scale)?.f_contiguous()?;
    }
    if let Some(table) = table {
        let searched = o.bits == 6 && o.search;
        if searched {
            let (r, u) = scale_search(rt, &gw, &rel, &ch, &table)?;
            rel = r;
            unsigned = Some(u);
        }
        if !searched || seed.positive(rt)? {
            unsigned = Some(assign_grid(
                rt,
                &gw,
                &levels(&table, &rel)?,
                &ch,
                seed,
                None,
            )?);
        }
        let mut u = unsigned.take().unwrap();
        if o.bits == 6 {
            let _ = u.f_add_scalar_(1)?;
        }
        unsigned = Some(u.f_view([n, k])?);
    }
    Ok((
        packing::pack_codes(&unsigned.unwrap(), o.bits)?,
        rel,
        ch.f_to_kind(Kind::Float)?.f_contiguous()?,
        correction,
        cb_out,
    ))
}
fn as_value(p: Prepared) -> Value {
    Value::Tuple(vec![
        Value::tensor(p.0),
        Value::tensor(p.1),
        Value::tensor(p.2),
        p.3.map_or(Value::None, Value::tensor),
        p.4.map_or(Value::None, Value::tensor),
    ])
}
fn prepare(rt: &Runtime, cuda: bool, w: &Tensor, mut o: Options) -> Result<Prepared, Error> {
    let (n, k) = (w.size()[0], w.size()[1]);
    let block = (BUDGET / k.max(1)).max(1);
    if o.bits == 6 {
        if !o.symmetric {
            return Err(fail("ValueError", "6-bit storage is symmetric only"));
        }
        o.codebook = false;
    }
    let cb = if o.symmetric && o.codebook {
        Some(if let Some(c) = o.override_cb.as_ref() {
            c.f_to_device_(w.device(), Kind::Float, false, false)?
        } else {
            decide(rt, cuda, w, o.group, o.rotation)?
        })
    } else {
        None
    };
    if n <= block {
        return rotated(
            rt,
            &rotate(rt, cuda, w, o.rotation)?,
            &o,
            cb.as_ref(),
            &o.seed,
        );
    }
    let (mut packed, mut rel, mut channel, mut corr) = (vec![], vec![], vec![], vec![]);
    let mut cb_out = None;
    for start in (0..n).step_by(block as usize) {
        let rot = rotate(
            rt,
            cuda,
            &w.f_narrow(0, start, block.min(n - start))?,
            o.rotation,
        )?;
        let p = rotated(rt, &rot, &o, cb.as_ref(), &o.seed.offset(rt, start)?)?;
        packed.push(p.0);
        rel.push(p.1);
        channel.push(p.2);
        if let Some(c) = p.3 {
            corr.push(c);
        }
        cb_out = p.4;
    }
    Ok((
        Tensor::f_cat(&packed, 0)?,
        Tensor::f_cat(&rel, 0)?,
        Tensor::f_cat(&channel, 0)?,
        if corr.is_empty() {
            None
        } else {
            Some(Tensor::f_cat(&corr, 1)?)
        },
        cb_out,
    ))
}
fn cuda_prepare(rt: &Runtime, w: &Tensor, mut o: Options) -> Result<Prepared, Error> {
    let (n, k) = (w.size()[0], w.size()[1]);
    let fused = o.scale == Kind::Float8e4m3fn
        && o.symmetric
        && ((o.bits == 4 && o.codebook && o.group == 16)
            || (o.bits == 6 && !o.search && [16, 32, 64].contains(&o.group)));
    if fused {
        let cb = if o.bits == 4 {
            Some(if let Some(c) = &o.override_cb {
                c.shallow_clone()
            } else {
                decide(rt, true, w, o.group, o.rotation)?
            })
        } else {
            None
        };
        if matches!(w.kind(), Kind::Half | Kind::BFloat16)
            && o.rotation == 256
            && k % 256 == 0
            && k % o.group == 0
        {
            let c = if let Some(c) = &cb {
                c.f_to_device_(w.device(), Kind::Float, false, false)?
                    .f_contiguous()?
            } else {
                crate::cuda_linear::empty(w.device(), Kind::Float)?
            };
            let q = Tensor::f_empty([n, k * o.bits / 8], (Kind::Int8, w.device()))?;
            let rel = Tensor::f_empty([n, k / o.group], (Kind::Float8e4m3fn, w.device()))?;
            let ch = Tensor::f_empty([n], (Kind::Float, w.device()))?;
            use KernelArg::{Bool, Int, Tensor as T};
            let (enabled, seed) = o.seed.native(rt, "quantize_wxa8_convrot_fused")?;
            let ok = unsafe {
                rt.native.call_unchecked(
                    "quantize_wxa8_convrot_fused",
                    &[
                        T(&w.f_contiguous()?),
                        T(&c),
                        T(&q),
                        T(&rel.f_view_dtype(Kind::Uint8)?),
                        T(&ch),
                        Int(o.bits),
                        Int(o.group),
                        Bool(enabled),
                        Int(seed),
                        Int(stream(rt, w)?),
                    ],
                )?
            }
            .value
                != 0;
            if ok {
                return Ok((q, rel, ch, None, if o.bits == 4 { Some(c) } else { None }));
            }
        }
        if o.bits == 4
            && matches!(w.kind(), Kind::Float | Kind::Half | Kind::BFloat16)
            && k <= 16 * (47 * 1024 / 4)
        {
            let cb = cb
                .unwrap()
                .f_to_device_(w.device(), Kind::Float, false, false)?
                .f_contiguous()?;
            let q = Tensor::f_empty([n, k / 2], (Kind::Int8, w.device()))?;
            let rel = Tensor::f_empty([n, k / 16], (Kind::Float8e4m3fn, w.device()))?;
            let ch = Tensor::f_empty([n], (Kind::Float, w.device()))?;
            let block = (BUDGET / k.max(1)).max(1);
            use KernelArg::{Bool, Int, Tensor as T};
            for start in (0..n).step_by(block as usize) {
                let count = block.min(n - start);
                let rot = rotate(rt, true, &w.f_narrow(0, start, count)?, o.rotation)?;
                let seed = o.seed.offset(rt, start)?;
                let (enabled, seed) = seed.native(rt, "quantize_w4a8_convrot")?;
                unsafe {
                    rt.native.call_unchecked(
                        "quantize_w4a8_convrot",
                        &[
                            T(&rot.f_contiguous()?),
                            T(&cb),
                            T(&q.f_narrow(0, start, count)?),
                            T(&rel.f_narrow(0, start, count)?.f_view_dtype(Kind::Uint8)?),
                            T(&ch.f_narrow(0, start, count)?),
                            Bool(enabled),
                            Int(seed),
                            Int(stream(rt, &rot)?),
                        ],
                    )?;
                }
            }
            return Ok((q, rel, ch, None, Some(cb)));
        }
        if cb.is_some() {
            o.override_cb = cb;
        }
    }
    prepare(rt, true, w, o)
}
fn run(rt: &Runtime, cuda: bool, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let w = tensor(a, kw, 0, "weight")?;
    let o = Options {
        group: integer(get(a, kw, 1, "group_size"), 16)?,
        rotation: integer(get(a, kw, 2, "convrot_groupsize"), 256)?,
        symmetric: boolean(get(a, kw, 3, "symmetric"), true)?,
        scale: dtype(get(a, kw, 4, "scale_dtype"), Kind::Float8e4m3fn)?,
        codebook: boolean(get(a, kw, 5, "codebook"), true)?,
        override_cb: optional_tensor(a, kw, 6, "codebook_tensor")?,
        seed: Seed::Raw(
            get(a, kw, 7, "stochastic_rounding")
                .cloned()
                .unwrap_or(Value::Int(0)),
        ),
        bits: integer(get(a, kw, 8, "bits"), 4)?,
        search: boolean(get(a, kw, 9, "scale_search"), true)?,
    };
    if w.dim() != 2 {
        return Err(fail(
            "ValueError",
            format!(
                "W4A8 weight must be 2D, got shape {}",
                super::w4a8::tuple(&w.size())
            ),
        ));
    }
    if ![4, 6].contains(&o.bits) {
        return Err(fail(
            "ValueError",
            format!("bits must be 4 or 6, got {}", o.bits),
        ));
    }
    let k = w.size()[1];
    if o.bits == 6 {
        packing::check_six_bit_layout(k, o.group)?;
    }
    if k % 16 == 0 && (o.group == 0 || (k % o.group == 0 && o.rotation == 0)) {
        return Err(fail("ZeroDivisionError", "integer modulo by zero"));
    }
    if k % 16 != 0
        || k % o.group != 0
        || k % o.rotation != 0
        || o.group < 4
        || (16 % o.group != 0 && o.group % 16 != 0)
    {
        return Err(fail(
            "ValueError",
            format!(
                "K={k} must be divisible by 16, group_size={}, and convrot_groupsize={}; group_size must be >=4 and divide 16 or be a multiple of 16",
                o.group, o.rotation
            ),
        ));
    }
    let prepared = if cuda {
        cuda_prepare(rt, &w, o)?
    } else {
        prepare(rt, false, &w, o)?
    };
    let original = get(a, kw, 6, "codebook_tensor");
    let reuse = match (original, prepared.4.as_ref()) {
        (Some(Value::Tensor(_)), Some(cb)) => {
            let source = original.unwrap().as_tensor()?;
            source.data_ptr() == cb.data_ptr()
                && source.kind() == cb.kind()
                && source.device() == cb.device()
                && source.size() == cb.size()
                && source.stride() == cb.stride()
        }
        _ => false,
    };
    let mut value = as_value(prepared);
    if reuse && let Value::Tuple(v) = &mut value {
        v[4] = original.unwrap().clone();
    }
    Ok(value)
}
pub(super) fn install(rt: &Runtime) {
    for backend in ["eager", "cuda"] {
        rt.backends.bind_reference(
            backend,
            "quantize_w4a8_int8_weight",
            body(move |rt, a, k| run(rt, backend == "cuda", a, k)),
        );
    }
}
