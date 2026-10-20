//! Shared CUDA geometry wrappers from Kitchen 0.2.38; original native kernels.
use super::*;
use tch::{Device, Kind, Tensor};
static MESH_BVH_CLASS: crate::execution_value::NamedTupleClass =
    crate::execution_value::NamedTupleClass { name: "MeshBVH" };
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(super) fn shape(name: &str, t: &Tensor, wanted: &[Option<i64>]) -> Result<(), Error> {
    if t.dim() != wanted.len()
        || t.size()
            .iter()
            .zip(wanted)
            .any(|(n, w)| w.is_some_and(|w| w != *n))
    {
        return Err(error(
            "ValueError",
            format!(
                "{name} must have shape ({}), got {}",
                wanted
                    .iter()
                    .map(|s| s.map_or("*".into(), |s| s.to_string()))
                    .collect::<Vec<_>>()
                    .join(", "),
                w4a8::tuple(&t.size())
            ),
        ));
    }
    Ok(())
}
fn count(rt: &Runtime, name: &str, value: &Value) -> Result<i64, Error> {
    match value {
        Value::Int(n) if *n >= 0 => Ok(*n),
        Value::UInt(n) => {
            i64::try_from(*n).map_err(|_| error("OverflowError", "int too big to convert"))
        }
        value => Err(error(
            "ValueError",
            format!(
                "{name} must be a non-negative integer, got {}",
                crate::python_value::repr_with_native(value, &rt.native)?
            ),
        )),
    }
}
fn indices(name: &str, index: &Tensor, n: i64, lowest: i64) -> Result<(), Error> {
    if index.numel() == 0 || crate::native::current_stream_capturing()? {
        return Ok(());
    }
    let (lo, hi) = index.f_aminmax(None, false)?;
    let values = Tensor::f_stack(&[lo, hi], 0)?.f_to_device(Device::Cpu)?;
    let lo = values.int64_value(&[0]);
    let hi = values.int64_value(&[1]);
    if lo < lowest || hi >= n {
        return Err(error(
            "ValueError",
            format!("{name} must lie in [{lowest}, {n}), got values in [{lo}, {hi}]"),
        ));
    }
    Ok(())
}
fn attr(value: &Value, name: &str) -> Result<Value, Error> {
    if let Value::NamedTuple(v) = value
        && let Some((_, v)) = v.fields.iter().find(|(field, _)| *field == name)
    {
        return Ok(v.clone());
    }
    Err(error(
        "AttributeError",
        format!(
            "'{}' object has no attribute '{name}'",
            crate::backend_triton_fp8::typename(value)
        ),
    ))
}
fn morton(x: &Tensor) -> Result<Tensor, Error> {
    let mut x = x.f_bitwise_and(0x1F_FFFF)?;
    for (shift, mask) in [
        (32, 0x001F_0000_0000_FFFF_i64),
        (16, 0x001F_0000_FF00_00FF),
        (8, 0x100F_00F0_0F00_F00F),
        (4, 0x10C3_0C30_C30C_30C3),
    ] {
        x = x
            .f_bitwise_or_tensor(&x.f_bitwise_left_shift_tensor_scalar(shift)?)?
            .f_bitwise_and(mask)?;
    }
    Ok(
        x.f_bitwise_or_tensor(&x.f_bitwise_left_shift_tensor_scalar(2)?)?
            .f_bitwise_and(0x1249_2492_4924_9249)?,
    )
}
fn bvh(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let triangles = tensor(a, kw, 0, "triangles")?;
    shape("triangles", &triangles, &[None, Some(3), Some(3)])?;
    let n = triangles.size()[0];
    let device = triangles.device();
    if n == 0 {
        return Err(error("ValueError", "mesh_bvh needs at least one triangle"));
    }
    let centre = triangles
        .f_amin([1], false)?
        .f_add(&triangles.f_amax([1], false)?)?
        .f_mul_scalar(0.5)?;
    let finite = centre.f_isfinite()?;
    let lo = centre
        .f_where_self(
            &finite,
            &Tensor::full([], f64::INFINITY, (centre.kind(), device)),
        )?
        .f_amin([0], false)?;
    let hi = centre
        .f_where_self(
            &finite,
            &Tensor::full([], f64::NEG_INFINITY, (centre.kind(), device)),
        )?
        .f_amax([0], false)?;
    let cell = centre
        .f_sub(&lo)?
        .f_div(&hi.f_sub(&lo)?.f_clamp_min(1e-30)?)?
        .f_mul_scalar((1 << 21) - 1)?
        .f_to_kind(Kind::Int64)?
        .f_clamp(0, (1 << 21) - 1)?;
    let codes = morton(&cell.f_select(1, 0)?)?
        .f_bitwise_left_shift_tensor_scalar(2)?
        .f_bitwise_or_tensor(
            &morton(&cell.f_select(1, 1)?)?.f_bitwise_left_shift_tensor_scalar(1)?,
        )?
        .f_bitwise_or_tensor(&morton(&cell.f_select(1, 2)?)?)?;
    let (codes, order) = codes.f_sort(-1, false)?;
    let tris = triangles.f_index(&[Some(&order)])?;
    let child = Tensor::f_empty([(n - 1).max(1), 2], (Kind::Int, device))?;
    let parent = Tensor::f_empty([2 * n - 1], (Kind::Int, device))?;
    let visits = Tensor::f_zeros([(n - 1).max(1)], (Kind::Int, device))?;
    let boxes = Tensor::f_empty([2 * n - 1, 6], (Kind::Float, device))?;
    use KernelArg::{Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            "mesh_bvh_build",
            &[
                T(&codes),
                T(&tris),
                Int(n),
                T(&child),
                T(&parent),
                T(&visits),
                T(&boxes),
                Int(stream(rt, &triangles)?),
            ],
        )?;
    }
    Ok(Value::named_tuple(
        &MESH_BVH_CLASS,
        vec![
            ("triangles", Value::tensor(tris)),
            ("index", Value::tensor(order.f_to_kind(Kind::Int)?)),
            ("child", Value::tensor(child.f_narrow(0, 0, n - 1)?)),
            ("box", Value::tensor(boxes)),
        ],
    ))
}
fn closest(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let bvh = get(a, kw, 0, "bvh").unwrap();
    let points = tensor(a, kw, 1, "points")?;
    shape("points", &points, &[None, Some(3)])?;
    let triangles = attr(bvh, "triangles")?.as_tensor()?;
    if triangles.device() != points.device() {
        return Err(error(
            "ValueError",
            format!(
                "bvh is on {} but points are on {}",
                crate::python_value::display(&Value::Device(triangles.device()))?,
                crate::python_value::display(&Value::Device(points.device()))?
            ),
        ));
    }
    let n = points.size()[0];
    let points = points.f_contiguous()?;
    let cutoff = get(a, kw, 2, "max_dist").unwrap_or(&Value::Float(f64::INFINITY));
    let max_dist = crate::python_numbers::float_value(rt, cutoff)?;
    if max_dist < 0. {
        return Ok(Value::Tuple(vec![
            Value::tensor(Tensor::f_full(
                [n],
                max_dist,
                (Kind::Float, points.device()),
            )?),
            Value::tensor(crate::native::clone_tensor(&points)?),
            Value::tensor(Tensor::f_full([n], -1, (Kind::Int64, points.device()))?),
        ]));
    }
    let dist = Tensor::f_empty([n], (Kind::Float, points.device()))?;
    let nearest = Tensor::f_empty([n, 3], (Kind::Float, points.device()))?;
    let face = Tensor::f_empty([n], (Kind::Int64, points.device()))?;
    use KernelArg::{Float, Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            "closest_point_on_mesh",
            &[
                T(&points),
                Int(n),
                T(&triangles),
                T(&attr(bvh, "index")?.as_tensor()?),
                Int(triangles.size()[0]),
                T(&attr(bvh, "child")?.as_tensor()?),
                T(&attr(bvh, "box")?.as_tensor()?),
                Float(max_dist),
                T(&dist),
                T(&nearest),
                T(&face),
                Int(stream(rt, &points)?),
            ],
        )?;
    }
    Ok(Value::Tuple(vec![
        Value::tensor(dist),
        Value::tensor(nearest),
        Value::tensor(face),
    ]))
}
fn components(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let n = count(rt, "num_nodes", get(a, kw, 1, "num_nodes").unwrap())?;
    let edges = tensor(a, kw, 0, "edges")?;
    shape("edges", &edges, &[None, Some(2)])?;
    indices("edges", &edges, n, 0)?;
    let edges = edges.f_contiguous()?;
    let parent = Tensor::f_arange(n, (Kind::Int, edges.device()))?;
    let labels = Tensor::f_empty([n], (Kind::Int64, edges.device()))?;
    use KernelArg::{Bool, Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            "connected_components",
            &[
                T(&edges),
                Int(edges.size()[0]),
                Bool(edges.kind() == Kind::Int64),
                Int(n),
                T(&parent),
                T(&labels),
                Int(stream(rt, &edges)?),
            ],
        )?;
    }
    Ok(Value::tensor(labels))
}
fn collapse(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let vertices = tensor(a, kw, 0, "vertices")?;
    let faces = tensor(a, kw, 1, "faces")?;
    let edges = tensor(a, kw, 2, "edges")?;
    let positions = tensor(a, kw, 3, "positions")?;
    let nv = *vertices
        .size()
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let ne = *edges
        .size()
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    shape("vertices", &vertices, &[None, Some(3)])?;
    shape("faces", &faces, &[None, Some(3)])?;
    shape("edges", &edges, &[None, Some(2)])?;
    shape("positions", &positions, &[Some(ne), Some(3)])?;
    indices("faces", &faces, nv, 0)?;
    indices("edges", &edges, nv, 0)?;
    let faces = faces.f_to_kind(Kind::Int)?.f_contiguous()?;
    let corners = faces.f_reshape([-1])?;
    let device = vertices.device();
    let offsets = Tensor::f_zeros([nv + 1], (Kind::Int, device))?;
    let counts = corners.f_bincount(None::<&Tensor>, nv)?;
    let target = offsets.f_narrow(0, 1, nv)?;
    // out= has the original int32 destination cast, rather than widening the CSR.
    let _ = counts.f_cumsum_out(&target, 0, None)?;
    let cursor = Tensor::f_zeros([nv], (Kind::Int, device))?;
    let fan = Tensor::f_empty([corners.numel() as i64], (Kind::Int, device))?;
    let flips = Tensor::f_empty([ne], (Kind::Int, device))?;
    let skinny = Tensor::f_empty([ne], (Kind::Float, device))?;
    let link = Tensor::f_empty([ne], (Kind::Bool, device))?;
    let threshold = crate::python_numbers::float_value(
        rt,
        get(a, kw, 4, "cos_threshold").unwrap_or(&Value::Float(0.)),
    )?;
    use KernelArg::{Float, Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            "edge_collapse_checks",
            &[
                T(&vertices.f_contiguous()?),
                T(&faces),
                T(&offsets),
                T(&cursor),
                T(&fan),
                T(&edges.f_to_kind(Kind::Int)?.f_contiguous()?),
                T(&positions.f_contiguous()?),
                Int(corners.numel() as i64),
                Int(ne),
                Float(threshold),
                T(&flips),
                T(&skinny),
                T(&link),
                Int(stream(rt, &vertices)?),
            ],
        )?;
    }
    Ok(Value::Tuple(vec![
        Value::tensor(flips),
        Value::tensor(skinny),
        Value::tensor(link),
    ]))
}
fn min_cut(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let nbr = tensor(a, kw, 0, "nbr")?;
    let cap = tensor(a, kw, 1, "cap")?;
    let s = tensor(a, kw, 2, "s_cap")?;
    let t = tensor(a, kw, 3, "t_cap")?;
    let device = nbr.device();
    let dims = nbr.size();
    if dims.len() != 2 {
        return Err(error(
            "ValueError",
            if dims.len() < 2 {
                format!(
                    "not enough values to unpack (expected 2, got {})",
                    dims.len()
                )
            } else {
                "too many values to unpack (expected 2)".into()
            },
        ));
    }
    let (n, k) = (dims[0], dims[1]);
    if k > 128 {
        return Err(error(
            "ValueError",
            format!("nbr can have at most 128 slots per node (reverse slots are int8), got {k}"),
        ));
    }
    shape("cap", &cap, &[Some(n), Some(k)])?;
    shape("s_cap", &s, &[Some(n)])?;
    shape("t_cap", &t, &[Some(n)])?;
    indices("nbr", &nbr, n, -1)?;
    if n == 0 {
        return Ok(Value::tensor(Tensor::f_empty([0], (Kind::Bool, device))?));
    }
    let nbr = nbr.f_to_kind(Kind::Int)?.f_contiguous()?;
    let cap = cap.f_contiguous()?;
    let r = cap.f_empty_like()?;
    let rev = Tensor::f_empty([n, k], (Kind::Int8, device))?;
    let e = crate::native::clone_tensor(&s)?;
    let terminal = crate::native::clone_tensor(&t)?;
    let h = Tensor::f_empty([n], (Kind::Int, device))?;
    let lists = [
        Tensor::f_empty([n], (Kind::Int, device))?,
        Tensor::f_empty([n], (Kind::Int, device))?,
    ];
    let counts = [
        Tensor::f_zeros([1], (Kind::Int, device))?,
        Tensor::f_zeros([1], (Kind::Int, device))?,
    ];
    let active = Tensor::f_zeros([1], (Kind::Int, device))?;
    let queued = Tensor::f_zeros([n], (Kind::Int, device))?;
    let inc = Tensor::f_zeros([n, k], (Kind::Float, device))?;
    let hn = Tensor::f_empty([n], (Kind::Int, device))?;
    let bfs = Tensor::f_empty([2, n], (Kind::Int, device))?;
    let bfs_counts = Tensor::f_zeros([2], (Kind::Int, device))?;
    let tol = if cap.numel() > 0 {
        1e-12 * cap.f_max()?.double_value(&[])
    } else {
        0.
    };
    let big = n + 2;
    let stream = stream(rt, &nbr)?;
    use KernelArg::{Float, Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            "min_cut_init",
            &[
                Int(n),
                Int(k),
                T(&nbr),
                T(&cap),
                T(&r),
                T(&rev),
                Int(stream),
            ],
        )?;
    }
    let relabel = get(a, kw, 4, "relabel_every").unwrap_or(&Value::Int(64));
    let relabel = integer(Some(relabel), 64)?;
    let mut cur = 0;
    let mut it = 0;
    loop {
        if relabel == 0 {
            return Err(error("ZeroDivisionError", "integer modulo by zero"));
        }
        if it % relabel == 0 {
            unsafe {
                rt.native.call_unchecked(
                    "min_cut_global_relabel",
                    &[
                        Int(n),
                        Int(k),
                        T(&nbr),
                        T(&rev),
                        T(&r),
                        T(&terminal),
                        Float(tol),
                        Int(big),
                        T(&h),
                        T(&bfs),
                        T(&bfs_counts),
                        Int(stream),
                    ],
                )?;
                rt.native.call_unchecked(
                    "min_cut_rebuild_active",
                    &[
                        Int(n),
                        T(&e),
                        T(&h),
                        Float(tol),
                        Int(big),
                        T(&lists[cur]),
                        T(&counts[cur]),
                        T(&queued),
                        Int(stream),
                    ],
                )?;
            }
            if counts[cur].int64_value(&[0]) == 0 {
                break;
            }
        }
        unsafe {
            rt.native.call_unchecked(
                "min_cut_iterate",
                &[
                    Int(n),
                    Int(k),
                    T(&nbr),
                    T(&rev),
                    T(&r),
                    T(&e),
                    T(&terminal),
                    T(&h),
                    T(&inc),
                    T(&hn),
                    Float(tol),
                    Int(big),
                    T(&lists[cur]),
                    T(&counts[cur]),
                    T(&lists[1 - cur]),
                    T(&counts[1 - cur]),
                    T(&queued),
                    T(&active),
                    Int(stream),
                ],
            )?;
        }
        cur = 1 - cur;
        it += 1;
        if active.int64_value(&[0]) == 0 {
            break;
        }
    }
    unsafe {
        rt.native.call_unchecked(
            "min_cut_global_relabel",
            &[
                Int(n),
                Int(k),
                T(&nbr),
                T(&rev),
                T(&r),
                T(&terminal),
                Float(tol),
                Int(big),
                T(&h),
                T(&bfs),
                T(&bfs_counts),
                Int(stream),
            ],
        )?;
    }
    Ok(Value::tensor(h.f_ge(big)?))
}
pub(super) fn install(rt: &Runtime) {
    for (name, f) in [
        (
            "mesh_bvh",
            bvh as fn(&Runtime, &[Value], &Kwargs) -> Result<Value, Error>,
        ),
        ("closest_point_on_mesh", closest),
        ("connected_components", components),
        ("edge_collapse_checks", collapse),
        ("min_cut", min_cut),
    ] {
        rt.backends.bind_reference("cuda", name, body(f));
    }
}
