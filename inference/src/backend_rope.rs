//! Frozen eager/CUDA RoPE bodies and exact inference-only storage checks.
use super::*;

pub(super) fn install(rt: &Runtime) {
    for backend in ["eager", "cuda"] {
        let cuda = backend == "cuda";
        for op in [
            "apply_rope1",
            "apply_rope1_",
            "apply_rope",
            "apply_rope_",
            "apply_rope_split_half1",
            "apply_rope_split_half1_",
            "apply_rope_split_half",
            "apply_rope_split_half_",
            "rms_rope1",
            "rms_rope1_",
            "rms_rope",
            "rms_rope_",
            "rms_rope_split_half1",
            "rms_rope_split_half1_",
            "rms_rope_split_half",
            "rms_rope_split_half_",
        ] {
            rt.backends
                .bind_reference(backend, op, body(move |rt, a, k| run(rt, cuda, op, a, k)));
        }
    }
}
fn error(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn trim(x: &Tensor, f: &Tensor) -> Result<Tensor, Error> {
    let xs = x.size();
    let fs = f.size();
    Ok(
        if xs.len() > 2 && fs.len() > 2 && xs[2] != 1 && fs[2] > xs[2] {
            f.f_slice(2, 0, xs[2], 1)?
        } else {
            f.shallow_clone()
        },
    )
}
fn bounds(t: &Tensor, offset: i64, bytes: i64) -> (i128, i128) {
    let (mut lo, mut hi) = (offset as i128, offset as i128);
    for (s, st) in t.size().iter().zip(t.stride()) {
        let extent = (*s as i128 - 1) * st as i128;
        lo += extent.min(0);
        hi += extent.max(0);
    }
    (lo * bytes as i128, (hi + 1) * bytes as i128 - 1)
}
fn interleaved(x: &Tensor, y: &Tensor, xoffset: i64, yoffset: i64) -> bool {
    let shape = x.size();
    let strides = x.stride();
    if shape != y.size() || strides != y.stride() || strides.last() != Some(&1) {
        return false;
    }
    let mut extent = 1_i128;
    let mut i = strides.len();
    while i > 0 && strides[i - 1] as i128 == extent {
        extent *= shape[i - 1] as i128;
        i -= 1;
    }
    let outer: Vec<_> = shape[..i]
        .iter()
        .zip(&strides[..i])
        .filter_map(|(n, s)| (*n > 1).then_some(*s))
        .collect();
    let Some(period) = outer.iter().min().copied() else {
        return false;
    };
    if period <= 0 || outer.iter().any(|s| s % period != 0) {
        return false;
    }
    let delta = (xoffset as i128 - yoffset as i128).abs();
    delta >= extent && delta + extent <= period as i128
}
fn overlap(rt: &Runtime, x: &Tensor, y: &Tensor) -> Result<bool, Error> {
    if x.numel() == 0 || y.numel() == 0 || x.device() != y.device() {
        return Ok(false);
    }
    let xi = rt.native.tensor_storage(x)?;
    let yi = rt.native.tensor_storage(y)?;
    if xi.base != yi.base {
        return Ok(false);
    }
    let (x0, x1) = bounds(x, xi.offset, xi.element_bytes);
    let (y0, y1) = bounds(y, yi.offset, yi.element_bytes);
    if x0.max(y0) > x1.min(y1) {
        return Ok(false);
    }
    Ok(!interleaved(x, y, xi.offset, yi.offset))
}
pub(crate) fn inplace_check(
    rt: &Runtime,
    xs: &[&Tensor],
    readonly: &[&Tensor],
) -> Result<(), Error> {
    if xs.iter().chain(readonly.iter()).any(|t| t.requires_grad()) {
        return Err(error(
            "RuntimeError",
            "in-place RoPE operations are inference-only and do not support autograd",
        ));
    }
    for x in xs {
        let mut dimensions: Vec<_> = x
            .size()
            .into_iter()
            .zip(x.stride())
            .filter_map(|(n, s)| (n > 1).then_some((s.abs(), n)))
            .collect();
        dimensions.sort();
        let mut required = 1_i128;
        for (stride, size) in dimensions {
            if (stride as i128) < required {
                return Err(error(
                    "ValueError",
                    "in-place RoPE requires views without internal overlap",
                ));
            }
            required = stride as i128 * size as i128;
        }
    }
    if xs.len() == 2 && overlap(rt, xs[0], xs[1])? {
        return Err(error(
            "ValueError",
            "paired in-place RoPE requires non-overlapping input storage",
        ));
    }
    for x in xs {
        for source in readonly {
            if overlap(rt, x, source)? {
                return Err(error(
                    "ValueError",
                    "in-place RoPE inputs must not overlap frequencies or scales",
                ));
            }
        }
    }
    Ok(())
}
fn eager_apply(x: &Tensor, f: &Tensor, split: bool) -> Result<Tensor, Error> {
    let shape = x.size();
    let out = if !split {
        let mut newshape = shape[..shape.len().saturating_sub(1)].to_vec();
        newshape.extend([-1, 1, 2]);
        let input = x.f_to_kind(f.kind())?.f_reshape(newshape.as_slice())?;
        let f = trim(x, f)?;
        let mut out = f.f_select(-1, 0)?.f_mul(&input.f_select(-1, 0)?)?;
        let _ = out.f_addcmul_(&f.f_select(-1, 1)?, &input.f_select(-1, 1)?)?;
        out
    } else {
        // promote_types for the three original allowed floating input dtypes.
        let promoted = if x.kind() == f.kind() {
            x.kind()
        } else if x.kind() == Kind::Float
            || f.kind() == Kind::Float
            || matches!(
                (x.kind(), f.kind()),
                (Kind::Half, Kind::BFloat16) | (Kind::BFloat16, Kind::Half)
            )
        {
            Kind::Float
        } else {
            return Err(Error::BindingGap {
                backend: "eager".into(),
                operation: "nonstandard RoPE dtype promotion".into(),
            });
        };
        let t = if promoted != f.kind() {
            x.f_to_kind(f.kind())?
        } else {
            x.shallow_clone()
        };
        let f = if f.dim() == 2 {
            f.f_unsqueeze(0)?
        } else {
            f.shallow_clone()
        };
        let diagonal = f.f_diagonal(0, -2, -1)?.f_movedim_int(-1, -2)?;
        let out = t.f_unflatten(-1, [2, -1])?.f_mul(&diagonal)?;
        let half = *shape
            .last()
            .ok_or_else(|| error("IndexError", "tuple index out of range"))?
            / 2;
        let mut first = out.f_select(-2, 0)?;
        let mut second = out.f_select(-2, 1)?;
        let _ = first.f_addcmul_(
            &t.f_slice(-1, half, i64::MAX, 1)?,
            &f.f_select(-2, 0)?.f_select(-1, 1)?,
        )?;
        let _ = second.f_addcmul_(
            &t.f_slice(-1, 0, half, 1)?,
            &f.f_select(-2, 1)?.f_select(-1, 0)?,
        )?;
        out
    };
    Ok(out.f_reshape(shape.as_slice())?.f_to_kind(x.kind())?)
}
fn eager_rms(
    x: &Tensor,
    f: &Tensor,
    s: &Tensor,
    eps: f64,
    split: bool,
    rot: i64,
) -> Result<Tensor, Error> {
    let d = *x
        .size()
        .last()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let normalized = x.f_rms_norm([d], Some(s), eps)?;
    if rot != 0 && rot != d {
        let rotated = eager_apply(&normalized.f_slice(-1, 0, rot, 1)?, f, split)?;
        return Ok(Tensor::f_cat(
            &[rotated, normalized.f_slice(-1, rot, i64::MAX, 1)?],
            -1,
        )?);
    }
    eager_apply(&normalized, f, split)
}
fn native_layout(x: &Tensor, f: &Tensor, rot: i64) -> Option<bool> {
    let xs = x.size();
    let fs = f.size();
    if xs.len() != 4 || fs.len() != 6 {
        return None;
    }
    let d = xs[3];
    let r = if rot != 0 { rot } else { d };
    if d == 0
        || d % 2 != 0
        || r % 2 != 0
        || r > d
        || fs[3..] != [r / 2, 2, 2]
        || ![1, xs[0]].contains(&fs[0])
    {
        return None;
    }
    let layout = if fs[1] == 1 && [1, xs[2]].contains(&fs[2]) {
        Some(false)
    } else if [1, xs[1]].contains(&fs[1]) && fs[2] == 1 {
        Some(true)
    } else {
        None
    };
    if d < 32 || d % 32 != 0 || !matches!(x.kind(), Kind::Half | Kind::BFloat16) {
        None
    } else {
        layout
    }
}
fn native_apply(
    rt: &Runtime,
    x: &Tensor,
    k: Option<&Tensor>,
    f: &Tensor,
    split: bool,
    inplace: bool,
) -> Result<(Tensor, Option<Tensor>), Error> {
    use KernelArg::{Bool, Int, None as Nil, Tensor as T};
    let f = if split {
        f.shallow_clone()
    } else {
        trim(x, f)?
    };
    let out = if inplace {
        x.shallow_clone()
    } else {
        rt.native.empty_like(x, x.device())?
    };
    let kout = k
        .map(|k| {
            if inplace {
                Ok(k.shallow_clone())
            } else {
                rt.native.empty_like(k, k.device())
            }
        })
        .transpose()?;
    unsafe {
        rt.native.call_unchecked(
            "apply_rope",
            &[
                T(x),
                T(&f),
                T(&out),
                k.map_or(Nil, T),
                kout.as_ref().map_or(Nil, T),
                Int(stream(rt, x)?),
                Bool(split),
            ],
        )?;
    }
    Ok((out, kout))
}
struct Rms<'a> {
    x: &'a Tensor,
    k: Option<&'a Tensor>,
    f: &'a Tensor,
    scale: &'a Tensor,
    kscale: Option<&'a Tensor>,
    eps: f64,
    split: bool,
    inplace: bool,
    rot: i64,
}
fn native_rms(rt: &Runtime, r: &Rms<'_>) -> Result<(Tensor, Option<Tensor>), Error> {
    use KernelArg::{Bool, Float, Int, Tensor as T};
    if native_layout(r.x, r.f, r.rot).is_none() {
        return Err(error("AssertionError", ""));
    }
    let out = if r.inplace {
        r.x.shallow_clone()
    } else {
        rt.native.empty_like(r.x, r.x.device())?
    };
    if let Some(k) = r.k {
        let kout = if r.inplace {
            k.shallow_clone()
        } else {
            rt.native.empty_like(k, k.device())?
        };
        unsafe {
            rt.native.call_unchecked(
                "rms_rope",
                &[
                    T(r.x),
                    T(k),
                    T(r.f),
                    T(r.scale),
                    T(r.kscale.unwrap_or(r.scale)),
                    T(&out),
                    T(&kout),
                    Float(r.eps),
                    Int(stream(rt, r.x)?),
                    Bool(r.split),
                    Int(r.rot),
                ],
            )?;
        }
        Ok((out, Some(kout)))
    } else {
        unsafe {
            rt.native.call_unchecked(
                "rms_rope1",
                &[
                    T(r.x),
                    T(r.f),
                    T(r.scale),
                    T(&out),
                    Float(r.eps),
                    Int(stream(rt, r.x)?),
                    Bool(r.split),
                ],
            )?;
        }
        Ok((out, None))
    }
}
fn run(rt: &Runtime, cuda: bool, op: &str, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let single = op.ends_with('1') || op.ends_with("1_");
    let inplace = op.ends_with('_');
    let rms = op.starts_with("rms_");
    let split = op.contains("split_half");
    let xname = if single {
        "x"
    } else if rms {
        "q"
    } else {
        "xq"
    };
    let kname = if rms { "k" } else { "xk" };
    let xvalue = required(a, kw, 0, xname)?;
    let x = xvalue.as_tensor()?;
    let kvalue = if single {
        None
    } else {
        Some(required(a, kw, 1, kname)?)
    };
    let k = kvalue.map(Value::as_tensor).transpose()?;
    let fidx = if single { 1 } else { 2 };
    let f = tensor(a, kw, fidx, "freqs_cis")?;
    let sidx = fidx + 1;
    let scale = if rms {
        Some(tensor(
            a,
            kw,
            sidx,
            if single { "scale" } else { "q_scale" },
        )?)
    } else {
        None
    };
    let kscale = if rms && !single {
        optional_tensor(a, kw, sidx + 1, "k_scale")?
    } else {
        None
    };
    let eps = if rms {
        real(get(a, kw, if single { 3 } else { 5 }, "epsilon"), 1e-6)?
    } else {
        0.
    };
    let rot = if rms && split && !single {
        integer(get(a, kw, 6, "rot_dim"), 0)?
    } else {
        0
    };
    if inplace {
        let mut xs = vec![&x];
        if let Some(k) = &k {
            xs.push(k);
        }
        let mut readonly = vec![&f];
        if let Some(s) = &scale {
            readonly.push(s);
        }
        if rms && !single {
            readonly.push(kscale.as_ref().unwrap_or(scale.as_ref().unwrap()));
        }
        inplace_check(rt, &xs, &readonly)?;
    }
    let (out, kout) = if !cuda {
        let first = if rms {
            eager_rms(&x, &f, scale.as_ref().unwrap(), eps, split, rot)?
        } else {
            eager_apply(&x, &f, split)?
        };
        let second = k
            .as_ref()
            .map(|k| {
                if rms {
                    eager_rms(
                        k,
                        &f,
                        kscale.as_ref().unwrap_or(scale.as_ref().unwrap()),
                        eps,
                        split,
                        rot,
                    )
                } else {
                    eager_apply(k, &f, split)
                }
            })
            .transpose()?;
        if inplace {
            let mut dst = x.shallow_clone();
            dst.f_copy_(&first)?;
            if let (Some(k), Some(out)) = (&k, &second) {
                let mut dst = k.shallow_clone();
                dst.f_copy_(out)?;
            }
        }
        (first, second)
    } else if !rms {
        if k.as_ref().is_some_and(|k| k.size() != x.size()) {
            let (first, _) = native_apply(rt, &x, None, &f, split, inplace)?;
            let (second, _) = native_apply(rt, k.as_ref().unwrap(), None, &f, split, inplace)?;
            (first, Some(second))
        } else {
            native_apply(rt, &x, k.as_ref(), &f, split, inplace)?
        }
    } else if k.as_ref().is_some_and(|k| k.size() != x.size()) {
        let k = k.as_ref().unwrap();
        let s = scale.as_ref().unwrap();
        let ks = kscale.as_ref().unwrap_or(s);
        if rot != 0 {
            // This eager fallback is present at this exact branch in CUDA.
            let first = eager_rms(&x, &f, s, eps, split, rot)?;
            let second = eager_rms(k, &f, ks, eps, split, rot)?;
            if inplace {
                let mut x = x.shallow_clone();
                x.f_copy_(&first)?;
                let mut k = k.shallow_clone();
                k.f_copy_(&second)?;
            }
            (first, Some(second))
        } else {
            let (first, _) = native_rms(
                rt,
                &Rms {
                    x: &x,
                    k: None,
                    f: &f,
                    scale: s,
                    kscale: None,
                    eps,
                    split,
                    inplace,
                    rot: 0,
                },
            )?;
            let (second, _) = native_rms(
                rt,
                &Rms {
                    x: k,
                    k: None,
                    f: &f,
                    scale: ks,
                    kscale: None,
                    eps,
                    split,
                    inplace,
                    rot: 0,
                },
            )?;
            (first, Some(second))
        }
    } else {
        native_rms(
            rt,
            &Rms {
                x: &x,
                k: k.as_ref(),
                f: &f,
                scale: scale.as_ref().unwrap(),
                kscale: kscale.as_ref(),
                eps,
                split,
                inplace,
                rot,
            },
        )?
    };
    Ok(if single {
        if inplace {
            xvalue.clone()
        } else {
            Value::tensor(out)
        }
    } else {
        Value::Tuple(if inplace {
            vec![xvalue.clone(), kvalue.unwrap().clone()]
        } else {
            vec![Value::tensor(out), Value::tensor(kout.unwrap())]
        })
    })
}
