//! Frozen eager tile/group SDPA algorithm and original fused CUDA NA3d wrapper.
use super::*;
use std::collections::HashMap;

#[path = "backend_na_cuda.rs"]
mod cuda_values;
#[path = "backend_na_dynamic.rs"]
mod dynamic;
#[path = "backend_na_values.rs"]
mod values;
const SCORE_BUDGET: i128 = 1 << 25;
const KV_STACK_BUDGET: i64 = 1 << 28;
type Bounds = (Vec<i64>, Vec<i64>);
type Region = [(i64, i64); 3];
type Tile = (Region, Region);

fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn shape(t: &Tensor) -> Result<[i64; 6], Error> {
    t.size().try_into().map_err(|_| {
        error(
            "ValueError",
            if t.dim() < 6 {
                format!("not enough values to unpack (expected 6, got {})", t.dim())
            } else {
                "too many values to unpack (expected 6)".into()
            },
        )
    })
}
fn sequence(v: &Value) -> Result<&[Value], Error> {
    match v {
        Value::List(v) | Value::Tuple(v) => Ok(v),
        _ => Err(Error::BindingGap {
            backend: "na3d".into(),
            operation: "dynamic iterable argument".into(),
        }),
    }
}
fn truth(v: &Value) -> Result<bool, Error> {
    Ok(match v {
        Value::None => false,
        Value::Bool(v) => *v,
        Value::Int(v) => *v != 0,
        Value::Float(v) => *v != 0.,
        Value::Text(v) => !v.is_empty(),
        Value::List(v) | Value::Tuple(v) => !v.is_empty(),
        Value::Dict(v) => !v.is_empty(),
        _ => {
            return Err(Error::BindingGap {
                backend: "na3d".into(),
                operation: "dynamic causal truth conversion".into(),
            });
        }
    })
}
fn scale(v: Option<&Value>, hd: i64) -> Result<f64, Error> {
    match v {
        None | Some(Value::None) if hd == 0 => Err(error(
            "ZeroDivisionError",
            "0.0 cannot be raised to a negative power",
        )),
        None | Some(Value::None) => Ok((hd as f64).powf(-0.5)),
        Some(Value::Bool(v)) => Ok(i64::from(*v) as f64),
        v => real(v, 1.),
    }
}
fn window_bounds(length: i64, kernel: i64, causal: bool) -> Bounds {
    let mut starts = vec![];
    let mut ends = vec![];
    let kernel = if causal { kernel } else { kernel.min(length) };
    for i in 0..length {
        let start = if causal {
            0.max(i - kernel + 1)
        } else {
            (i - kernel / 2).max(0).min(length - kernel)
        };
        starts.push(start);
        ends.push(if causal { i + 1 } else { start + kernel });
    }
    (starts, ends)
}
fn pick_tiles(dims: [i64; 3], kernels: [i64; 3]) -> [i64; 3] {
    let mut tiles = dims;
    let cost = |ts: [i64; 3]| {
        let nq: i128 = ts.into_iter().map(i128::from).product();
        let nk: i128 = (0..3)
            .map(|i| i128::from(dims[i].min(ts[i] + kernels[i] - 1)))
            .product();
        nq * nk
    };
    while cost(tiles) > SCORE_BUDGET && *tiles.iter().max().unwrap() > 1 {
        let mut axis = 0;
        for a in 1..3 {
            // Python max retains the first axis on a tie.
            if tiles[a] as f64 / kernels[a] as f64 > tiles[axis] as f64 / kernels[axis] as f64 {
                axis = a;
            }
        }
        if tiles[axis] <= 1 {
            break;
        }
        tiles[axis] = 1.max((tiles[axis] + 1) / 2);
    }
    tiles
}
fn region(t: &Tensor, r: Region) -> Result<Tensor, Error> {
    let mut t = t.shallow_clone();
    for (i, (start, end)) in r.into_iter().enumerate() {
        t = t.f_slice(i as i64 + 1, start, end, 1)?;
    }
    Ok(t)
}
fn minimum(kind: Kind) -> Result<f64, Error> {
    Ok(match kind {
        Kind::Half | Kind::ComplexHalf => -65504.,
        Kind::BFloat16 => -3.3895313892515355e38,
        Kind::Float | Kind::ComplexFloat => f32::MIN as f64,
        Kind::Double | Kind::ComplexDouble => f64::MIN,
        _ => {
            return Err(Error::BindingGap {
                backend: "eager".into(),
                operation: "na3d torch.finfo dynamic dtype".into(),
            });
        }
    })
}
fn mask(rel: &[Bounds; 3], kind: Kind, dev: tch::Device) -> Result<Tensor, Error> {
    let mut bools = vec![];
    for (starts, ends) in rel {
        let st = Tensor::f_from_slice(starts)?.f_to_device(dev)?;
        let en = Tensor::f_from_slice(ends)?.f_to_device(dev)?;
        // The original int(en.max()) performs the tensor reduction and host read,
        // including current-stream synchronization on CUDA. Do not replace it
        // with the equal host-list maximum as an unapproved optimization.
        let length = en.f_max()?.f_int64_value(&[])?;
        let kj = Tensor::f_arange(length, (Kind::Int64, dev))?.f_unsqueeze(0)?;
        bools.push(
            kj.f_ge_tensor(&st.f_unsqueeze(1)?)?
                .f_bitwise_and_tensor(&kj.f_lt_tensor(&en.f_unsqueeze(1)?)?)?,
        );
    }
    let nq: [i64; 3] = std::array::from_fn(|i| bools[i].size()[0]);
    let nk: [i64; 3] = std::array::from_fn(|i| bools[i].size()[1]);
    let visible = bools[0]
        .f_reshape([nq[0], 1, 1, nk[0], 1, 1])?
        .f_bitwise_and_tensor(&bools[1].f_reshape([1, nq[1], 1, 1, nk[1], 1])?)?
        .f_bitwise_and_tensor(&bools[2].f_reshape([1, 1, nq[2], 1, 1, nk[2]])?)?;
    let min = minimum(kind)?;
    let (nq, nk) = (
        nq.into_iter().product::<i64>(),
        nk.into_iter().product::<i64>(),
    );
    let mut out = Tensor::f_zeros([nq, nk], (kind, dev))?;
    let _ = out.f_masked_fill_(&visible.f_reshape([nq, nk])?.f_bitwise_not()?, min)?;
    Ok(out.f_reshape([1, 1, nq, nk])?)
}
fn eager(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dims: [i64; 6],
    kernels: [i64; 3],
    causal: [bool; 3],
    scale: f64,
) -> Result<Value, Error> {
    let [batch, t, h, w, nh, hd] = dims;
    let axes = [t, h, w];
    let kernels: [i64; 3] = std::array::from_fn(|i| {
        if causal[i] {
            kernels[i]
        } else {
            kernels[i].min(axes[i])
        }
    });
    let q = if scale != 1. {
        q.f_mul_scalar(scale)?
    } else {
        q
    };
    let bounds = std::array::from_fn::<_, 3, _>(|i| window_bounds(axes[i], kernels[i], causal[i]));
    let tile = pick_tiles(axes, std::array::from_fn(|i| kernels[i].min(axes[i])));
    if tile.contains(&0) {
        return Err(error("ValueError", "range() arg 3 must not be zero"));
    }
    let mut indices = HashMap::<[Bounds; 3], usize>::new();
    let mut groups: Vec<([Bounds; 3], Vec<Tile>)> = vec![];
    for t0 in (0..t).step_by(tile[0] as usize) {
        for h0 in (0..h).step_by(tile[1] as usize) {
            for w0 in (0..w).step_by(tile[2] as usize) {
                let starts = [t0, h0, w0];
                let qs = std::array::from_fn(|i| (starts[i], axes[i].min(starts[i] + tile[i])));
                let rs = std::array::from_fn(|i| {
                    (
                        bounds[i].0[qs[i].0 as usize],
                        bounds[i].1[qs[i].1 as usize - 1],
                    )
                });
                let rel = std::array::from_fn(|i| {
                    let (start, end) = qs[i];
                    let relative = |v: &[i64]| {
                        v[start as usize..end as usize]
                            .iter()
                            .map(|s| s - rs[i].0)
                            .collect()
                    };
                    (relative(&bounds[i].0), relative(&bounds[i].1))
                });
                let index = *indices.entry(rel.clone()).or_insert_with(|| {
                    groups.push((rel, vec![]));
                    groups.len() - 1
                });
                groups[index].1.push((qs, rs));
            }
        }
    }
    let out = Tensor::f_empty(dims, (v.kind(), q.device()))?;
    for (rel, tiles) in groups {
        let mask = mask(&rel, q.kind(), q.device())?;
        let (nq, nk) = (mask.size()[2], mask.size()[3]);
        let gmax = if q.device().is_cuda() {
            1.max(KV_STACK_BUDGET / 1.max(batch * nh * nk * hd * 2))
        } else {
            1
        };
        let sizes = tiles[0].0.map(|(s, e)| e - s);
        for chunk in tiles.chunks(gmax as usize) {
            let g = chunk.len() as i64;
            let stack = |x: &Tensor, query: bool, n: i64| -> Result<Tensor, Error> {
                let regions = chunk
                    .iter()
                    .map(|(qs, rs)| region(x, if query { *qs } else { *rs }))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Tensor::f_stack(&regions, 0)?
                    .f_permute([0, 1, 5, 2, 3, 4, 6])?
                    .f_reshape([g * batch, nh, n, hd])?)
            };
            let qs = stack(&q, true, nq)?;
            let ks = stack(&k, false, nk)?;
            let vs = stack(&v, false, nk)?;
            let o = Tensor::f_scaled_dot_product_attention(
                &qs,
                &ks,
                &vs,
                Some(&mask),
                0.,
                false,
                Some(1.),
                false,
            )?
            .f_view_([g, batch, nh, sizes[0], sizes[1], sizes[2], hd].as_slice())?
            .f_permute([0, 1, 3, 4, 5, 2, 6])?;
            for (i, (qs, _)) in chunk.iter().enumerate() {
                region(&out, *qs)?.f_copy_(&o.f_get(i as i64)?)?;
            }
        }
    }
    Ok(Value::tensor(out))
}
fn run(rt: &Runtime, cuda: bool, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    if cuda {
        return cuda_values::run(rt, a, k);
    }
    if !cuda {
        fn plain_sequence(v: Option<&Value>, optional: bool) -> bool {
            match v {
                None | Some(Value::None) if optional => true,
                Some(Value::List(v) | Value::Tuple(v)) => v
                    .iter()
                    .all(|v| matches!(v, Value::Int(_) | Value::Bool(_))),
                _ => false,
            }
        }
        let plain_scale = matches!(
            get(a, k, 5, "scale"),
            None | Some(Value::None | Value::Bool(_) | Value::Int(_) | Value::Float(_))
        );
        if !plain_sequence(get(a, k, 3, "kernel_size"), false)
            || !plain_sequence(get(a, k, 4, "is_causal"), true)
            || !plain_scale
        {
            return dynamic::run(rt, a, k);
        }
    }
    let q = tensor(a, k, 0, "q")?;
    let key = tensor(a, k, 1, "k")?;
    let v = tensor(a, k, 2, "v")?;
    let dims = shape(&q)?;
    let causal = match get(a, k, 4, "is_causal") {
        None | Some(Value::None) => vec![false; 3],
        Some(v) => sequence(v)?
            .iter()
            .map(truth)
            .collect::<Result<Vec<_>, _>>()?,
    };
    let kernels = sequence(required(a, k, 3, "kernel_size")?)?;
    if !cuda {
        // zip(kernel_size, causal, dims, strict=True), including first exhaustion.
        let n = kernels.len().min(causal.len()).min(3);
        let message = if n == kernels.len() {
            if causal.len() > n {
                Some("zip() argument 2 is longer than argument 1")
            } else if 3 > n {
                Some("zip() argument 3 is longer than arguments 1-2")
            } else {
                None
            }
        } else if n == causal.len() {
            Some("zip() argument 2 is shorter than argument 1")
        } else {
            Some("zip() argument 3 is shorter than arguments 1-2")
        };
        if let Some(message) = message {
            return Err(error("ValueError", message));
        }
    }
    let kernels = [
        integer(kernels.first(), 0)?,
        integer(kernels.get(1), 0)?,
        integer(kernels.get(2), 0)?,
    ];
    let causal = [causal[0], causal[1], causal[2]];
    eager(
        q,
        key,
        v,
        dims,
        kernels,
        causal,
        scale(get(a, k, 5, "scale"), dims[5])?,
    )
}
pub(super) fn install(rt: &Runtime) {
    for backend in ["eager", "cuda"] {
        rt.backends.bind_reference(
            backend,
            "na3d",
            body(move |rt, a, k| run(rt, backend == "cuda", a, k)),
        );
    }
}
