//! Original eager algorithm for source values retained in window geometry.
use super::values::{Number as N, array};
use super::*;
type Bounds = (Vec<N>, Vec<N>);
type DynamicRegion = [(N, N); 3];
type Tile = (Region, DynamicRegion);
fn bounds(rt: &Runtime, length: i64, kernel: &N, causal: bool) -> Result<Bounds, Error> {
    let mut starts = vec![];
    let mut ends = vec![];
    if causal {
        for i in 0..length {
            starts.push(
                N::Int(0).max(
                    rt,
                    &N::Int(i128::from(i)).binary(rt, kernel, "sub")?.binary(
                        rt,
                        &N::Int(1),
                        "add",
                    )?,
                )?,
            );
            ends.push(N::Int(i128::from(i) + 1));
        }
    } else {
        let kernel = kernel.min(rt, &N::Int(i128::from(length)))?;
        let low = N::Int(i128::from(length)).binary(rt, &kernel, "sub")?;
        let half = kernel.binary(rt, &N::Int(2), "floordiv")?;
        for i in 0..length {
            let start = N::Int(i128::from(i))
                .binary(rt, &half, "sub")?
                .max(rt, &N::Int(0))?
                .min(rt, &low)?;
            ends.push(start.binary(rt, &kernel, "add")?);
            starts.push(start);
        }
    }
    Ok((starts, ends))
}
fn pick(rt: &Runtime, dims: [i64; 3], kernels: &[N; 3]) -> Result<[i64; 3], Error> {
    let mut tiles = dims;
    let cost = |tiles: [i64; 3]| -> Result<N, Error> {
        let nq = tiles.into_iter().map(i128::from).product::<i128>();
        let mut nk = N::Int(1);
        for axis in 0..3 {
            let len = N::Int(i128::from(tiles[axis]))
                .binary(rt, &kernels[axis], "add")?
                .binary(rt, &N::Int(1), "sub")?;
            nk = nk.binary(rt, &N::Int(i128::from(dims[axis])).min(rt, &len)?, "mul")?;
        }
        N::Int(nq).binary(rt, &nk, "mul")
    };
    while N::Int(SCORE_BUDGET).lt(rt, &cost(tiles)?)? && *tiles.iter().max().unwrap() > 1 {
        let mut axis = 0;
        for i in 1..3 {
            let a = N::Int(i128::from(tiles[axis])).binary(rt, &kernels[axis], "div")?;
            let b = N::Int(i128::from(tiles[i])).binary(rt, &kernels[i], "div")?;
            if a.lt(rt, &b)? {
                axis = i;
            }
        }
        if tiles[axis] <= 1 {
            break;
        }
        tiles[axis] = ((tiles[axis] + 1) / 2).max(1);
    }
    Ok(tiles)
}
fn mask(rt: &Runtime, rel: &[Bounds; 3], kind: Kind, device: tch::Device) -> Result<Tensor, Error> {
    let mut bools = vec![];
    for (starts, ends) in rel {
        let st = array(rt, starts, device)?;
        let en = array(rt, ends, device)?;
        // Preserve the original tensor reduction, host conversion and sync.
        let length = super::super::seeds::Seed::Raw(Value::tensor(en.f_max()?)).integer(rt)?;
        let length =
            i64::try_from(length).map_err(|_| error("OverflowError", "int too big to convert"))?;
        let kj = Tensor::f_arange(length, (Kind::Int64, device))?.f_unsqueeze(0)?;
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
    let min = super::minimum(kind)?;
    let nq = nq.into_iter().product::<i64>();
    let nk = nk.into_iter().product::<i64>();
    let mut mask = Tensor::f_zeros([nq, nk], (kind, device))?;
    let _ = mask.f_masked_fill_(&visible.f_reshape([nq, nk])?.f_bitwise_not()?, min)?;
    Ok(mask.f_reshape([1, 1, nq, nk])?)
}
fn dynamic_region(t: &Tensor, region: &DynamicRegion) -> Result<Tensor, Error> {
    let mut t = t.shallow_clone();
    for (axis, (start, end)) in region.iter().enumerate() {
        t = t.f_slice(axis as i64 + 1, start.index()?, end.index()?, 1)?;
    }
    Ok(t)
}
pub(super) fn run(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let q = tensor(a, kw, 0, "q")?;
    let key = tensor(a, kw, 1, "k")?;
    let v = tensor(a, kw, 2, "v")?;
    let [batch, t, h, w, nh, hd] = shape(&q)?;
    let dims = [t, h, w];
    let causal = match get(a, kw, 4, "is_causal") {
        None | Some(Value::None) => vec![Value::Bool(false); 3],
        Some(v) => crate::backend_triton_na::sequence(rt, v)?,
    };
    let raw = crate::backend_triton_na::sequence(rt, required(a, kw, 3, "kernel_size")?)?;
    let mut kernels = vec![];
    let mut flags = vec![];
    for (axis, (kernel, c)) in raw.iter().zip(&causal).take(3).enumerate() {
        let flag = crate::backend_triton_mxfp8::truthy(rt, c)?;
        let kernel = N::from_value(kernel);
        kernels.push(if flag {
            kernel
        } else {
            kernel.min(rt, &N::Int(i128::from(dims[axis])))?
        });
        flags.push(flag);
    }
    crate::backend_triton_na::zip_end(raw.len(), causal.len())?;
    let kernels: [N; 3] = kernels.try_into().unwrap_or_else(|_| unreachable!());
    let scale = match get(a, kw, 5, "scale") {
        None | Some(Value::None) => {
            if hd == 0 {
                return Err(error(
                    "ZeroDivisionError",
                    "0.0 cannot be raised to a negative power",
                ));
            }
            Value::Float((hd as f64).powf(-0.5))
        }
        Some(v) => v.clone(),
    };
    // For numeric values != 1.0 is exact. Other Tensor shapes retain their
    // original comparison and ambiguous-bool exception through ATen.
    let equal = match &scale {
        Value::Tensor(_) | Value::Quantized(_) => {
            let comparison = unsafe {
                rt.op(
                    "aten::ne.Scalar",
                    &[scale.clone(), Value::Float(1.)],
                    &vec![],
                )?
            };
            !crate::backend_triton_mxfp8::truthy(rt, &comparison)?
        }
        Value::Float(n) => *n == 1.,
        Value::Int(n) => *n == 1,
        Value::UInt(n) => *n == 1,
        Value::Bool(b) => *b,
        _ => false,
    };
    let q = if equal {
        q
    } else {
        match &scale {
            Value::Text(_) | Value::List(_) | Value::Tuple(_) => {
                return Err(type_error(
                    "only integer tensors of a single element can be converted to an index",
                ));
            }
            Value::Dict(_) | Value::DType(_) | Value::Device(_) => {
                return Err(type_error(format!(
                    "unsupported operand type(s) for *: 'Tensor' and '{}'",
                    crate::backend_triton_fp8::typename(&scale)
                )));
            }
            _ => {}
        }
        let q = Value::tensor(q);
        let op = if matches!(scale, Value::Tensor(_) | Value::Quantized(_)) {
            "aten::mul.Tensor"
        } else {
            "aten::mul.Scalar"
        };
        unsafe { rt.op(op, &[q, scale], &vec![])? }.as_tensor()?
    };
    let bounds = (0..3)
        .map(|i| bounds(rt, dims[i], &kernels[i], flags[i]))
        .collect::<Result<Vec<_>, _>>()?;
    let bounded = (0..3)
        .map(|i| kernels[i].min(rt, &N::Int(i128::from(dims[i]))))
        .collect::<Result<Vec<_>, _>>()?;
    let bounded: [N; 3] = bounded.try_into().unwrap_or_else(|_| unreachable!());
    let tile = pick(rt, dims, &bounded)?;
    if tile.contains(&0) {
        return Err(error("ValueError", "range() arg 3 must not be zero"));
    }
    let mut indices = HashMap::<Vec<Vec<String>>, usize>::new();
    let mut groups: Vec<([Bounds; 3], Vec<Tile>)> = vec![];
    for t0 in (0..t).step_by(tile[0] as usize) {
        for h0 in (0..h).step_by(tile[1] as usize) {
            for w0 in (0..w).step_by(tile[2] as usize) {
                let starts = [t0, h0, w0];
                let qs: Region =
                    std::array::from_fn(|i| (starts[i], dims[i].min(starts[i] + tile[i])));
                let rs: DynamicRegion = std::array::from_fn(|i| {
                    (
                        bounds[i].0[qs[i].0 as usize].clone(),
                        bounds[i].1[qs[i].1 as usize - 1].clone(),
                    )
                });
                let relative = (0..3)
                    .map(|i| -> Result<Bounds, Error> {
                        Ok((
                            bounds[i].0[qs[i].0 as usize..qs[i].1 as usize]
                                .iter()
                                .map(|s| s.binary(rt, &rs[i].0, "sub"))
                                .collect::<Result<_, _>>()?,
                            bounds[i].1[qs[i].0 as usize..qs[i].1 as usize]
                                .iter()
                                .map(|s| s.binary(rt, &rs[i].0, "sub"))
                                .collect::<Result<_, _>>()?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let rel: [Bounds; 3] = relative.try_into().unwrap_or_else(|_| unreachable!());
                let hash = rel
                    .iter()
                    .flat_map(|(starts, ends)| {
                        [starts, ends]
                            .into_iter()
                            .map(|values| values.iter().map(N::key).collect::<Vec<_>>())
                    })
                    .collect::<Vec<_>>();
                let index = *indices.entry(hash).or_insert_with(|| {
                    groups.push((rel, vec![]));
                    groups.len() - 1
                });
                groups[index].1.push((qs, rs));
            }
        }
    }
    let out = Tensor::f_empty([batch, t, h, w, nh, hd], (v.kind(), q.device()))?;
    for (rel, tiles) in groups {
        let mask = mask(rt, &rel, q.kind(), q.device())?;
        let (nq, nk) = (mask.size()[2], mask.size()[3]);
        let gmax = if matches!(q.device(), tch::Device::Cuda(_)) {
            (KV_STACK_BUDGET / (batch * nh * nk * hd * 2).max(1)).max(1)
        } else {
            1
        };
        let sizes = tiles[0].0.map(|(s, e)| e - s);
        for chunk in tiles.chunks(gmax as usize) {
            let g = chunk.len() as i64;
            let stack = |x: &Tensor, query: bool, n: i64| -> Result<Tensor, Error> {
                let regions = chunk
                    .iter()
                    .map(|(qs, rs)| {
                        if query {
                            region(x, *qs)
                        } else {
                            dynamic_region(x, rs)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Tensor::f_stack(&regions, 0)?
                    .f_permute([0, 1, 5, 2, 3, 4, 6])?
                    .f_reshape([g * batch, nh, n, hd])?)
            };
            let qs = stack(&q, true, nq)?;
            let ks = stack(&key, false, nk)?;
            let vs = stack(&v, false, nk)?;
            let output = Tensor::f_scaled_dot_product_attention(
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
                region(&out, *qs)?.f_copy_(&output.f_get(i as i64)?)?;
            }
        }
    }
    Ok(Value::tensor(out))
}
