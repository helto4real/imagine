//! Original eager/CUDA conv3d and per-frame GroupNorm/padding wrappers.
use super::*;
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use tch::Device;
type ZeroVectors = HashMap<(i64, Device, Kind), Tensor>;
static ZEROS: LazyLock<Mutex<ZeroVectors>> = LazyLock::new(|| Mutex::new(HashMap::new()));
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn seq(v: &Value) -> Result<Vec<i64>, Error> {
    match v {
        Value::List(xs) | Value::Tuple(xs) => xs.iter().map(|v| integer(Some(v), 0)).collect(),
        _ => Err(type_error("Expected integer sequence")),
    }
}
fn unpack<const N: usize>(v: &[i64]) -> Result<[i64; N], Error> {
    v.try_into().map_err(|_| {
        error(
            "ValueError",
            if v.len() < N {
                format!(
                    "not enough values to unpack (expected {N}, got {})",
                    v.len()
                )
            } else {
                format!("too many values to unpack (expected {N})")
            },
        )
    })
}
fn ints(xs: &[i64]) -> Value {
    Value::List(xs.iter().copied().map(Value::Int).collect())
}
fn aten(rt: &Runtime, name: &str, overload: &str, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    unsafe { crate::boxed_aten::call(&rt.native, name, overload, a, k) }
}
fn cl(rt: &Runtime, t: &Tensor) -> Result<Tensor, Error> {
    aten(
        rt,
        "aten::contiguous",
        "",
        &[Value::tensor(t.shallow_clone())],
        &vec![("memory_format".into(), Value::Int(3))],
    )?
    .as_tensor()
}
fn is_cl(rt: &Runtime, t: &Tensor) -> Result<bool, Error> {
    match aten(
        rt,
        "aten::is_contiguous",
        "memory_format",
        &[Value::tensor(t.shallow_clone()), Value::Int(3)],
        &vec![],
    )? {
        Value::Bool(b) => Ok(b),
        _ => Err(Error::Native("is_contiguous returned non-bool".into())),
    }
}
fn empty_cl(rt: &Runtime, shape: &[i64], kind: Kind, dev: Device) -> Result<Tensor, Error> {
    aten(
        rt,
        "aten::empty",
        "memory_format",
        &[ints(shape)],
        &vec![
            ("dtype".into(), Value::DType(kind)),
            ("device".into(), Value::Device(dev)),
            ("memory_format".into(), Value::Int(3)),
        ],
    )?
    .as_tensor()
}
fn zero(n: i64, dev: Device, kind: Kind) -> Result<Tensor, Error> {
    let mut z = ZEROS.lock().unwrap();
    if let Some(t) = z.get(&(n, dev, kind)) {
        return Ok(t.shallow_clone());
    }
    let t = Tensor::f_zeros([n], (kind, dev))?;
    z.insert((n, dev, kind), t.shallow_clone());
    Ok(t)
}
fn aligned(t: &Tensor) -> bool {
    (t.data_ptr() as usize).is_multiple_of(16)
}
fn pad(rt: &Runtime, x: &Tensor, p: &[i64], mode: &str) -> Result<Tensor, Error> {
    aten(
        rt,
        "aten::pad",
        "",
        &[
            Value::tensor(x.shallow_clone()),
            ints(p),
            Value::Text(mode.into()),
        ],
        &vec![],
    )?
    .as_tensor()
}
fn opt(t: Option<&Tensor>) -> Value {
    t.map(|t| Value::tensor(t.shallow_clone()))
        .unwrap_or(Value::None)
}
fn conv(
    rt: &Runtime,
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    s: &[i64],
) -> Result<Tensor, Error> {
    aten(
        rt,
        "aten::conv3d",
        "",
        &[
            Value::tensor(x.shallow_clone()),
            Value::tensor(w.shallow_clone()),
            opt(b),
            ints(s),
        ],
        &vec![],
    )?
    .as_tensor()
}
fn ndhwc(t: &Tensor) -> Option<[i64; 4]> {
    let sz = t.size();
    if sz.len() < 2 {
        return None;
    }
    let (n, c) = (sz[0], sz[1]);
    let st = t.stride();
    if t.dim() != 5 || st[1] != 1 || st[4] != c {
        return None;
    }
    let s = [st[4], st[3], st[2], if n == 1 { 0 } else { st[0] }];
    if s.iter().any(|s| s % 8 != 0 || *s > i32::MAX as i64) {
        None
    } else {
        Some(s)
    }
}
#[allow(clippy::too_many_arguments)]
fn cutlass_conv(
    rt: &Runtime,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    res: Option<&Tensor>,
    stride: &[i64],
    out: Option<&Tensor>,
) -> Result<Option<Tensor>, Error> {
    use KernelArg::{Int, Tensor as T};
    let [n, c, d, h, ww] = unpack::<5>(&x.size())?;
    let [k, wc, t, r, s] = unpack::<5>(&w.size())?;
    let [sd, sh, sw] = unpack::<3>(stride)?;
    if sd.min(sh).min(sw) < 1 {
        return Ok(None);
    }
    let (z, p, q) = (
        (d - t).div_euclid(sd) + 1,
        (h - r).div_euclid(sh) + 1,
        (ww - s).div_euclid(sw) + 1,
    );
    let shape = [n, k, z, p, q];
    let supported = x.kind() == Kind::Half
        && w.kind() == Kind::Half
        && matches!(x.device(), Device::Cuda(_))
        && w.device() == x.device()
        && wc == c
        && (c % 8 == 0 || c < 8)
        && k % 8 == 0
        && d >= t
        && h >= r
        && ww >= s
        && bias.is_none_or(|b| b.kind() == Kind::Half && b.size() == [k])
        && res.is_none_or(|r| r.kind() == Kind::Half && r.size() == shape);
    if !supported {
        return Ok(None);
    }
    let (mut x, mut w) = (x.shallow_clone(), w.shallow_clone());
    let mut c = c;
    if c < 8 {
        x = pad(rt, &x, &[0, 0, 0, 0, 0, 0, 0, 8 - c], "constant")?;
        w = pad(rt, &w, &[0, 0, 0, 0, 0, 0, 0, 8 - c], "constant")?;
        c = 8;
    }
    let mut xs = ndhwc(&x);
    if xs.is_none() || is_cl(rt, &x)? {
        x = cl(rt, &x)?;
        xs = Some([0; 4]);
    }
    let w = cl(rt, &w)?;
    let b = match bias {
        Some(b) => b.f_contiguous()?,
        None => zero(k, x.device(), Kind::Half)?,
    };
    let res = match res {
        Some(r) => cl(rt, r)?,
        None => zero(k, x.device(), Kind::Half)?,
    };
    let (out, os) = match out {
        None => (empty_cl(rt, &shape, Kind::Half, x.device())?, [0; 4]),
        Some(out) => {
            if out.size() != shape || out.kind() != Kind::Half || out.device() != x.device() {
                return Err(error(
                    "ValueError",
                    "fp16_conv3d: out must be an fp16 [N, K, Z, P, Q] tensor on x's device",
                ));
            }
            let os = if is_cl(rt, out)? {
                [0; 4]
            } else {
                let required = [k, q * k, p * q * k, if n == 1 { 0 } else { z * p * q * k }];
                match ndhwc(out) {
                    Some(os) if os == required => os,
                    _ => return Ok(None),
                }
            };
            (out.shallow_clone(), os)
        }
    };
    if ![&x, &w, &b, &res, &out].iter().all(|t| aligned(t)) {
        return Ok(None);
    }
    let mut args = vec![T(&x), T(&w), T(&b), T(&res), T(&out)];
    args.extend(
        [n, d, h, ww, c, k, t, r, s, sd, sh, sw, stream(rt, &x)?, -1]
            .into_iter()
            .map(Int),
    );
    args.extend(xs.unwrap().into_iter().chain(os).map(Int));
    let ok = unsafe { rt.native.call_unchecked("cutlass_fp16_conv3d", &args)? };
    Ok((ok.value != 0).then_some(out))
}
#[allow(clippy::too_many_arguments)]
fn eager_group(
    rt: &Runtime,
    x: &Tensor,
    w: Option<&Tensor>,
    bias: Option<&Tensor>,
    groups: i64,
    eps: f64,
    padding: &[i64],
    silu: bool,
    zero_pad: bool,
    stats: Option<&Tensor>,
) -> Result<Tensor, Error> {
    if padding
        .iter()
        .min()
        .ok_or_else(|| error("ValueError", "min() iterable argument is empty"))?
        < &0
    {
        return Err(error(
            "ValueError",
            "group_norm_silu_pad3d: padding must be non-negative",
        ));
    }
    let [b, c, t, h, ww] = unpack::<5>(&x.size())?;
    let mut y = x.shallow_clone();
    if let Some(w) = w {
        let w = w.f_to_kind(x.kind())?;
        let bias = bias.map(|b| b.f_to_kind(x.kind())).transpose()?;
        let v = y.f_permute([0, 2, 1, 3, 4])?.f_reshape([b * t, c, h, ww])?;
        if let Some(stats) = stats {
            // A short stats tensor would broadcast silently.
            if stats.size() != [b * t, groups, 2] {
                return Err(error(
                    "ValueError",
                    format!(
                        "group_norm_silu_pad3d: stats must be [{}, {groups}, 2]",
                        b * t
                    ),
                ));
            }
            // Per-(frame, channel) scale and shift in fp32, rounded once like
            // group_norm's output.
            if groups == 0 {
                return Err(error(
                    "ZeroDivisionError",
                    "integer division or modulo by zero",
                ));
            }
            let per = |i: i64| -> Result<Tensor, Error> {
                Ok(stats.f_select(-1, i)?.f_repeat_interleave_self_int(
                    c.div_euclid(groups),
                    Some(1),
                    None,
                )?)
            };
            let scale = per(1)?.f_mul(&w.f_to_kind(Kind::Float)?)?;
            let mut shift = per(0)?.f_neg()?.f_mul(&scale)?;
            if let Some(bias) = &bias {
                shift = shift.f_add(&bias.f_to_kind(Kind::Float)?)?;
            }
            y = shift
                .f_unsqueeze(-1)?
                .f_unsqueeze(-1)?
                .f_addcmul(
                    &v.f_to_kind(Kind::Float)?,
                    &scale.f_unsqueeze(-1)?.f_unsqueeze(-1)?,
                )?
                .f_to_kind(x.kind())?
                .f_view_([b, t, c, h, ww].as_slice())?
                .f_permute([0, 2, 1, 3, 4])?;
        } else {
            if groups == 0 {
                return Err(error(
                    "ZeroDivisionError",
                    "integer division or modulo by zero",
                ));
            }
            if (b * t * c).div_euclid(groups) * h * ww == 1 {
                return Err(error(
                    "ValueError",
                    format!(
                        "Expected more than 1 value per channel when training, got input size [{}, {c}, {h}, {ww}]",
                        b * t
                    ),
                ));
            }
            y = aten(
                rt,
                "aten::group_norm",
                "",
                &[
                    Value::tensor(v),
                    Value::Int(groups),
                    Value::tensor(w),
                    opt(bias.as_ref()),
                    Value::Float(eps),
                ],
                &vec![],
            )?
            .as_tensor()?
            .f_view_([b, t, c, h, ww].as_slice())?
            .f_permute([0, 2, 1, 3, 4])?;
        }
    }
    if silu {
        y = y.f_silu()?;
    }
    let [l, r, tp, bt, f] = unpack::<5>(padding)?;
    if l != 0 || r != 0 || tp != 0 || bt != 0 {
        y = pad(
            rt,
            &y,
            &[l, r, tp, bt, 0, 0],
            if zero_pad { "constant" } else { "reflect" },
        )?;
    }
    if f != 0 {
        y = pad(rt, &y, &[0, 0, 0, 0, f, 0], "constant")?;
    }
    let out = cl(rt, &y)?;
    if w.is_none() && !silu && padding.iter().all(|p| *p == 0) && is_cl(rt, x)? {
        rt.native.clone_tensor(&out)
    } else {
        Ok(out)
    }
}
fn packed(rt: &Runtime, t: &Tensor) -> Result<bool, Error> {
    if is_cl(rt, t)? {
        return Ok(true);
    }
    let [b, c, _, h, w] = unpack::<5>(&t.size())?;
    let s = t.stride();
    Ok(b == 1 && s[1] == 1 && s[4] == c && s[3] == w * c && s[2] == h * w * c)
}
struct Group<'a> {
    x: &'a Tensor,
    w: Option<&'a Tensor>,
    bias: Option<&'a Tensor>,
    groups: i64,
    eps: f64,
    padding: &'a [i64],
    silu: bool,
    zero_pad: bool,
    out: Option<&'a Tensor>,
    stats: Option<&'a Tensor>,
}
/// Elements between image rows when x is a single frame with packed NHWC rows
/// that sit farther apart than W * C (a row-interleaved view the kernels read in
/// place); None otherwise.
fn ndhwc_row_stride(x: &Tensor) -> Result<Option<i64>, Error> {
    let [b, c, t, h, w] = unpack::<5>(&x.size())?;
    let s = x.stride();
    Ok((t == 1
        && s[1] == 1
        && s[4] == c
        && s[3] > w * c
        && s[3] % 8 == 0
        && (b == 1 || s[0] == h * s[3]))
        .then_some(s[3]))
}
/// Kitchen `_storage_bounds`: the byte range a view's elements span.
fn storage_bounds(x: &Tensor) -> (i64, i64) {
    let (mut lo, mut hi) = (x.storage_offset(), x.storage_offset());
    for (size, stride) in x.size().into_iter().zip(x.stride()) {
        let extent = (size - 1) * stride;
        lo += extent.min(0);
        hi += extent.max(0);
    }
    let element = x.kind().elt_size_in_bytes() as i64;
    (lo * element, (hi + 1) * element - 1)
}
/// Kitchen `_interleaved_disjoint`: same-layout views whose rows sit side by side
/// within a shared period never share an element.
fn interleaved_disjoint(x: &Tensor, y: &Tensor) -> bool {
    let strides = x.stride();
    if x.size() != y.size() || strides != y.stride() {
        return false;
    }
    if strides.last() != Some(&1) {
        return false;
    }
    let shape = x.size();
    let (mut extent, mut i) = (1, strides.len());
    while i > 0 && strides[i - 1] == extent {
        extent *= shape[i - 1];
        i -= 1;
    }
    let outer = shape[..i]
        .iter()
        .zip(&strides[..i])
        .filter(|(size, _)| **size > 1)
        .map(|(_, s)| *s)
        .collect::<Vec<_>>();
    let Some(&period) = outer.iter().min() else {
        return false;
    };
    if period <= 0 || outer.iter().any(|s| s % period != 0) {
        return false;
    }
    let delta = (x.storage_offset() - y.storage_offset()).abs();
    delta >= extent && delta + extent <= period
}
/// Kitchen `_tensors_overlap`.
fn tensors_overlap(x: &Tensor, y: &Tensor) -> bool {
    if x.numel() == 0 || y.numel() == 0 || x.device() != y.device() {
        return false;
    }
    // The storage's data pointer, as untyped_storage().data_ptr() gives it.
    let base =
        |t: &Tensor| t.data_ptr() as i64 - t.storage_offset() * t.kind().elt_size_in_bytes() as i64;
    if base(x) != base(y) {
        return false;
    }
    let ((x0, x1), (y0, y1)) = (storage_bounds(x), storage_bounds(y));
    if x0.max(y0) > x1.min(y1) {
        return false;
    }
    !interleaved_disjoint(x, y)
}
fn cuda_group(rt: &Runtime, g: &Group<'_>) -> Result<Tensor, Error> {
    use KernelArg::{Bool, Float, Int, Tensor as T};
    let Group {
        x,
        w,
        bias,
        groups,
        eps,
        padding,
        silu,
        zero_pad,
        out,
        stats,
    } = *g;
    let [b, c, t, h, ww] = unpack::<5>(&x.size())?;
    let [l, r, tp, bt, f] = unpack::<5>(padding)?;
    if padding.iter().min().unwrap() < &0 {
        return Err(error(
            "ValueError",
            "group_norm_silu_pad3d: padding must be non-negative",
        ));
    }
    // The affine parameters may be fp32; every path wants x's dtype.
    let w = w
        .map(|w| w.f_to_kind(x.kind())?.f_contiguous())
        .transpose()?;
    let bias = if w.is_some() {
        Some(match bias {
            Some(b) => b.f_to_kind(x.kind())?.f_contiguous()?,
            None => zero(c, x.device(), x.kind())?,
        })
    } else {
        bias.map(Tensor::shallow_clone)
    };
    if let Some(stats) = stats
        && (stats.size() != [b * t, groups, 2]
            || stats.kind() != Kind::Float
            || stats.device() != x.device())
    {
        return Err(error(
            "ValueError",
            format!(
                "group_norm_silu_pad3d: stats must be float32 [{}, {groups}, 2] on {}",
                b * t,
                match x.device() {
                    Device::Cuda(i) => format!("cuda:{i}"),
                    other => crate::execution_value::device_name(other),
                }
            ),
        ));
    }
    let mut served = !(c % 8 != 0 || c > 2048);
    if served && w.is_some() {
        if groups == 0 {
            return Err(error("ZeroDivisionError", "integer modulo by zero"));
        }
        served = c % groups == 0 && groups <= 1024;
    }
    served = served && (zero_pad || (l.max(r) < ww && tp.max(bt) < h)) && b * (t + f) <= 65535;
    let mut x = x.shallow_clone();
    let mut row_stride = if served { ndhwc_row_stride(&x)? } else { None };
    if served && row_stride.is_none() {
        row_stride = Some(ww * c);
        x = cl(rt, &x)?;
    }
    // The kernels issue 16-byte vector loads.
    served = served && aligned(&x);
    if !served {
        let res = eager_group(
            rt,
            &x,
            w.as_ref(),
            bias.as_ref(),
            groups,
            eps,
            padding,
            silu,
            zero_pad,
            stats,
        )?;
        return match out {
            None => Ok(res),
            Some(o) => {
                let mut o = o.shallow_clone();
                o.f_copy_(&res)?;
                Ok(o)
            }
        };
    }
    // Frames are written in no particular order, so an out overlapping x goes via
    // a copy.
    let direct = match out {
        Some(o) => packed(rt, o)? && aligned(o) && !tensors_overlap(o, &x),
        None => false,
    };
    let dst = if direct {
        out.unwrap().shallow_clone()
    } else {
        empty_cl(
            rt,
            &[b, c, t + f, h + tp + bt, ww + l + r],
            x.kind(),
            x.device(),
        )?
    };
    let placeholder = cuda_linear::empty(x.device(), Kind::Float)?;
    let (w, bias, ws) = match w {
        None => (
            placeholder.shallow_clone(),
            placeholder.shallow_clone(),
            placeholder.shallow_clone(),
        ),
        Some(w) if stats.is_none() => {
            let chunks = (h * ww + 1023) / 1024;
            let ws = Tensor::f_empty(
                [2 * b * t * (chunks * c + groups)],
                (Kind::Float, x.device()),
            )?;
            (w, bias.unwrap(), ws)
        }
        Some(w) => (w, bias.unwrap(), placeholder.shallow_clone()),
    };
    let stats = match stats {
        Some(s) => s.f_contiguous()?,
        None => placeholder,
    };
    unsafe {
        rt.native.call_unchecked(
            "group_norm_silu_pad3d",
            &[
                T(&x),
                T(&w),
                T(&bias),
                T(&dst),
                T(&ws),
                T(&stats),
                Int(b),
                Int(c),
                Int(t),
                Int(h),
                Int(ww),
                Int(groups),
                Float(eps),
                Int(l),
                Int(r),
                Int(tp),
                Int(bt),
                Int(f),
                Bool(silu),
                Bool(zero_pad),
                Int(row_stride.unwrap()),
                Int(quantization::dtype_code(x.kind())?),
                Int(stream(rt, &x)?),
            ],
        )?;
    }
    match out {
        Some(o) if !direct => {
            let mut o = o.shallow_clone();
            o.f_copy_(&dst)?;
            Ok(o)
        }
        _ => Ok(dst),
    }
}
/// Per-frame GroupNorm (mean, rstd) of x [B, C, T, H, W] as float32
/// [B*T, groups, 2], eagerly.
fn eager_group_stats(x: &Tensor, groups: i64, eps: f64) -> Result<Tensor, Error> {
    let [b, _, t, _, _] = unpack::<5>(&x.size())?;
    let frames = x
        .f_permute([0, 2, 1, 3, 4])?
        .f_reshape([b * t, groups, -1])?
        .f_to_kind(Kind::Float)?;
    let (var, mean) = frames.f_var_mean_dim([-1i64].as_slice(), false, false)?;
    Ok(Tensor::f_stack(
        &[mean, var.f_add_scalar(eps)?.f_rsqrt()?],
        -1,
    )?)
}
fn cuda_group_stats(rt: &Runtime, x: &Tensor, groups: i64, eps: f64) -> Result<Tensor, Error> {
    use KernelArg::{Float, Int, Tensor as T};
    let [b, c, t, h, ww] = unpack::<5>(&x.size())?;
    let mut served = !(c % 8 != 0 || c > 2048);
    if served {
        if groups == 0 {
            return Err(error("ZeroDivisionError", "integer modulo by zero"));
        }
        served = c % groups == 0 && groups <= 1024 && b * t <= 65535;
    }
    let mut x = x.shallow_clone();
    let mut row_stride = if served { ndhwc_row_stride(&x)? } else { None };
    if served && row_stride.is_none() {
        row_stride = Some(ww * c);
        x = cl(rt, &x)?;
    }
    if !(served && aligned(&x)) {
        return eager_group_stats(&x, groups, eps);
    }
    let chunks = (h * ww + 1023) / 1024;
    let ws = Tensor::f_empty([2 * b * t * chunks * c], (Kind::Float, x.device()))?;
    let stats = Tensor::f_empty([b * t, groups, 2], (Kind::Float, x.device()))?;
    unsafe {
        rt.native.call_unchecked(
            "group_norm_stats3d",
            &[
                T(&x),
                T(&ws),
                T(&stats),
                Int(b),
                Int(c),
                Int(t),
                Int(h),
                Int(ww),
                Int(groups),
                Float(eps),
                Int(row_stride.unwrap()),
                Int(quantization::dtype_code(x.kind())?),
                Int(stream(rt, &x)?),
            ],
        )?;
    }
    Ok(stats)
}
pub(super) fn install(rt: &Runtime) {
    for backend in ["eager", "cuda"] {
        let cuda = backend == "cuda";
        for op in [
            "fp16_conv3d",
            "fp16_conv3d_out",
            "group_norm_silu_pad3d",
            "group_norm_silu_pad3d_out",
            "group_norm_stats3d",
        ] {
            rt.backends
                .bind_reference(backend, op, body(move |rt, a, k| run(rt, cuda, op, a, k)));
        }
    }
}
fn run(rt: &Runtime, cuda: bool, op: &str, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let x = tensor(a, k, 0, "x")?;
    if op == "group_norm_stats3d" {
        let groups = integer(get(a, k, 1, "num_groups"), 0)?;
        let eps = real(get(a, k, 2, "eps"), 0.)?;
        return Ok(Value::tensor(if cuda {
            cuda_group_stats(rt, &x, groups, eps)?
        } else {
            eager_group_stats(&x, groups, eps)?
        }));
    }
    let w = optional_tensor(a, k, 1, "weight")?;
    let b = optional_tensor(a, k, 2, "bias")?;
    let is_out = op.ends_with("_out");
    if op.starts_with("fp16_conv3d") {
        let w = w.ok_or_else(|| type_error("Expected weight Tensor"))?;
        let res = optional_tensor(a, k, 3, "residual")?;
        let s = seq(required(a, k, 4, "stride")?)?;
        let out = if is_out {
            Some(tensor(a, k, 5, "out")?)
        } else {
            None
        };
        if cuda
            && let Some(v) = cutlass_conv(rt, &x, &w, b.as_ref(), res.as_ref(), &s, out.as_ref())?
        {
            return Ok(if is_out {
                Value::None
            } else {
                Value::tensor(v)
            });
        }
        let mut v = conv(rt, &x, &w, b.as_ref(), &s)?;
        if cuda && !is_out {
            v = cl(rt, &v)?;
        }
        if let Some(r) = res {
            v = v.f_add(&r)?;
        }
        if let Some(mut out) = out {
            out.f_copy_(&v)?;
            Ok(Value::None)
        } else {
            Ok(Value::tensor(if cuda { v } else { cl(rt, &v)? }))
        }
    } else {
        let groups = integer(get(a, k, 3, "num_groups"), 0)?;
        let eps = real(get(a, k, 4, "eps"), 0.)?;
        let padding = seq(required(a, k, 5, "pad")?)?;
        let silu = boolean(get(a, k, 6, "silu"), false)?;
        let zp = boolean(get(a, k, 7, "zero_pad"), false)?;
        let ov = if cuda || is_out {
            get(a, k, 8, "out").filter(|v| !matches!(v, Value::None))
        } else {
            None
        };
        let ot = ov.map(Value::as_tensor).transpose()?;
        // stats follows out where the signature has one.
        let stats = get(a, k, if cuda || is_out { 9 } else { 8 }, "stats")
            .filter(|v| !matches!(v, Value::None))
            .map(Value::as_tensor)
            .transpose()?;
        let v = if cuda {
            cuda_group(
                rt,
                &Group {
                    x: &x,
                    w: w.as_ref(),
                    bias: b.as_ref(),
                    groups,
                    eps,
                    padding: &padding,
                    silu,
                    zero_pad: zp,
                    out: ot.as_ref(),
                    stats: stats.as_ref(),
                },
            )?
        } else {
            eager_group(
                rt,
                &x,
                w.as_ref(),
                b.as_ref(),
                groups,
                eps,
                &padding,
                silu,
                zp,
                stats.as_ref(),
            )?
        };
        if is_out {
            if !cuda {
                let mut out = ot.ok_or_else(|| type_error("Expected out Tensor"))?;
                out.f_copy_(&v)?;
            }
            Ok(Value::None)
        } else if let Some(ov) = ov {
            Ok(ov.clone())
        } else {
            Ok(Value::tensor(v))
        }
    }
}
