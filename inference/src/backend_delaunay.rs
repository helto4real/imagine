//! Kitchen's incremental Delaunay orchestration and unchanged native flip kernels.
use super::*;
use crate::sampling::locals::SourceLocals;
use tch::{Kind, Tensor};
const MAX_TETS: i64 = 1 << 29;
const INT_MAX: i64 = i32::MAX as i64;
struct State<'a> {
    rt: &'a Runtime,
    n: i64,
    cap: i64,
    nt: i64,
    stream: i64,
    verts: Tensor,
    v: Tensor,
    opp: Tensor,
    alive: Tensor,
    pend: Tensor,
    pt: Tensor,
    prio: Tensor,
    to_pt: Tensor,
    locals: SourceLocals,
}
impl State<'_> {
    fn empty(&self, n: i64) -> Result<Tensor, Error> {
        Ok(Tensor::f_empty([n], (Kind::Int, self.verts.device()))?)
    }
    fn zeros(&self, n: i64) -> Result<Tensor, Error> {
        Ok(Tensor::f_zeros([n], (Kind::Uint8, self.verts.device()))?)
    }
    fn grow(&mut self, need: i64) -> Result<(), Error> {
        if need <= self.cap {
            return Ok(());
        }
        if need > MAX_TETS {
            return Err(w4a8::err(
                "ValueError",
                format!(
                    "delaunay3d: {} points need more than {MAX_TETS} tets",
                    self.n
                ),
            ));
        }
        let extra = need.max(self.cap * 3 / 2).min(MAX_TETS) - self.cap;
        let device = self.verts.device();
        self.v = Tensor::f_cat(
            &[
                &self.v,
                &Tensor::f_full([extra, 4], -1, (Kind::Int, device))?,
            ],
            0,
        )?;
        self.opp = Tensor::f_cat(
            &[
                &self.opp,
                &Tensor::f_full([extra, 4], -1, (Kind::Int, device))?,
            ],
            0,
        )?;
        self.alive = Tensor::f_cat(&[&self.alive, &self.zeros(extra)?], 0)?;
        self.pend = Tensor::f_cat(&[&self.pend, &self.zeros(4 * extra)?], 0)?;
        self.cap += extra;
        Ok(())
    }
    fn compact(&self, rows: i64) -> Result<(Tensor, Tensor, Tensor), Error> {
        let new_index = self
            .alive
            .f_narrow(0, 0, self.nt)?
            .f_cumsum(0, Kind::Int)?
            .f_sub_scalar(1)?;
        let out_v = Tensor::f_full([rows, 4], -1, (Kind::Int, self.verts.device()))?;
        let out_o = Tensor::f_full([rows, 4], -1, (Kind::Int, self.verts.device()))?;
        use KernelArg::{Int, Tensor as T};
        unsafe {
            self.rt.native.call_unchecked(
                "delaunay_compact",
                &[
                    Int(self.nt),
                    T(&self.alive),
                    T(&new_index),
                    T(&self.v),
                    T(&self.opp),
                    T(&out_v),
                    T(&out_o),
                    Int(self.stream),
                ],
            )?;
        }
        Ok((new_index, out_v, out_o))
    }
    fn flip(&mut self, mut active: Tensor) -> Result<(), Error> {
        let mut locals = SourceLocals::default();
        for _ in 0..256 {
            let flag = active
                .f_narrow(0, 0, self.nt)?
                .f_mul(&self.alive.f_narrow(0, 0, self.nt)?)?;
            let _ = locals.keep("active_flag", flag.shallow_clone());
            let list = flag.f_nonzero()?.f_squeeze_dim(1)?.f_to_kind(Kind::Int)?;
            let _ = locals.keep("active_list", list.shallow_clone());
            if list.numel() == 0 {
                return Ok(());
            }
            let nc = 4 * list.numel() as i64;
            let info = self.empty(nc)?;
            let _ = locals.keep("cand_info", info.shallow_clone());
            let cn = self.empty(nc)?;
            let _ = locals.keep("cand_n", cn.shallow_clone());
            let cm = self.empty(nc)?;
            let _ = locals.keep("cand_m", cm.shallow_clone());
            let owner = Tensor::f_full([self.nt], INT_MAX, (Kind::Int, self.verts.device()))?;
            let _ = locals.keep("owner", owner.shallow_clone());
            active = self.zeros(self.cap)?;
            use KernelArg::{Int, Tensor as T};
            unsafe {
                self.rt.native.call_unchecked(
                    "delaunay_flip_detect",
                    &[
                        Int(list.numel() as i64),
                        T(&list),
                        T(&flag),
                        T(&self.v),
                        T(&self.opp),
                        T(&self.verts),
                        T(&info),
                        T(&cn),
                        T(&cm),
                        T(&owner),
                        T(&active),
                        Int(self.stream),
                    ],
                )?;
            }
            let run = self.empty(nc)?;
            let _ = locals.keep("run", run.shallow_clone());
            unsafe {
                self.rt.native.call_unchecked(
                    "delaunay_flip_decide",
                    &[
                        Int(nc),
                        T(&list),
                        T(&info),
                        T(&cn),
                        T(&cm),
                        T(&owner),
                        T(&run),
                        T(&active),
                        Int(self.stream),
                    ],
                )?;
            }
            let ran = run.f_ne(0)?;
            let _ = locals.keep("ran", ran.shallow_clone());
            let is23 = run.f_bitwise_and(3)?.f_eq(1)?;
            let _ = locals.keep("is23", is23.shallow_clone());
            let totals = Tensor::f_stack(&[ran.f_sum(Kind::Int64)?, is23.f_sum(Kind::Int64)?], 0)?
                .f_to_device(tch::Device::Cpu)?;
            let nr = totals.int64_value(&[0]);
            let n23 = totals.int64_value(&[1]);
            drop(totals);
            if nr == 0 {
                return Ok(());
            }
            let slot = is23
                .f_cumsum(0, Kind::Int)?
                .f_sub(&is23.f_to_kind(Kind::Int)?)?;
            let _ = locals.keep("slot_rank", slot.shallow_clone());
            let rank = ran
                .f_cumsum(0, Kind::Int)?
                .f_sub(&ran.f_to_kind(Kind::Int)?)?;
            let _ = locals.keep("run_rank", rank.shallow_clone());
            self.grow(self.nt + n23)?;
            active = Tensor::f_cat(
                &[&active, &self.zeros(self.cap - active.numel() as i64)?],
                0,
            )?;
            let flipped = Tensor::f_zeros([self.nt], (Kind::Int, self.verts.device()))?;
            let _ = locals.keep("flipped", flipped.shallow_clone());
            let redirect = self.empty(4 * self.nt)?;
            let _ = locals.keep("redirect", redirect.shallow_clone());
            let new_tets = self.empty(3 * nr)?;
            let _ = locals.keep("new_tets", new_tets.shallow_clone());
            unsafe {
                self.rt.native.call_unchecked(
                    "delaunay_flip_execute",
                    &[
                        Int(nc),
                        T(&list),
                        T(&run),
                        T(&cn),
                        T(&cm),
                        T(&slot),
                        T(&rank),
                        Int(self.nt),
                        T(&self.verts),
                        T(&self.v),
                        T(&self.opp),
                        T(&self.alive),
                        T(&flipped),
                        T(&redirect),
                        T(&self.pend),
                        T(&new_tets),
                        T(&active),
                        Int(self.stream),
                    ],
                )?;
                self.rt.native.call_unchecked(
                    "delaunay_flip_fixup",
                    &[
                        Int(nr),
                        T(&new_tets),
                        T(&flipped),
                        T(&redirect),
                        T(&self.opp),
                        T(&self.pend),
                        Int(self.stream),
                    ],
                )?;
                self.rt.native.call_unchecked(
                    "delaunay_relocate_flip",
                    &[
                        Int(self.n),
                        T(&flipped),
                        T(&new_tets),
                        T(&self.v),
                        T(&self.verts),
                        T(&self.pt),
                        Int(self.stream),
                    ],
                )?;
            }
            self.nt += n23;
        }
        Ok(())
    }
    fn insert_all(&mut self) -> Result<(), Error> {
        loop {
            let vote = Tensor::f_full([self.nt], INT_MAX, (Kind::Int, self.verts.device()))?;
            let _ = self.locals.keep("vote", vote.shallow_clone());
            use KernelArg::{Int, Tensor as T};
            unsafe {
                self.rt.native.call_unchecked(
                    "delaunay_vote",
                    &[
                        T(&self.pt),
                        T(&self.prio),
                        Int(self.n),
                        T(&vote),
                        Int(self.stream),
                    ],
                )?;
            }
            let split = vote.f_ne(INT_MAX)?;
            let _ = self.locals.keep("split", split.shallow_clone());
            let ns = split.f_sum(Kind::Int64)?.int64_value(&[]);
            if ns == 0 {
                break;
            }
            let rank = split
                .f_cumsum(0, Kind::Int)?
                .f_sub(&split.f_to_kind(Kind::Int)?)?;
            let _ = self.locals.keep("split_rank", rank.shallow_clone());
            self.grow(self.nt + 3 * ns)?;
            let active = self.zeros(self.cap)?;
            let _ = self.locals.keep("active", active.shallow_clone());
            unsafe {
                self.rt.native.call_unchecked(
                    "delaunay_split",
                    &[
                        Int(self.nt),
                        T(&vote),
                        T(&rank),
                        T(&self.to_pt),
                        T(&self.v),
                        T(&self.opp),
                        T(&self.alive),
                        T(&self.pt),
                        T(&active),
                        Int(self.stream),
                    ],
                )?;
                self.rt.native.call_unchecked(
                    "delaunay_relocate_split",
                    &[
                        Int(self.n),
                        T(&vote),
                        T(&rank),
                        Int(self.nt),
                        T(&self.v),
                        T(&self.verts),
                        T(&self.pt),
                        Int(self.stream),
                    ],
                )?;
            }
            self.nt += 3 * ns;
            self.flip(active)?;
            let alive = self
                .alive
                .f_narrow(0, 0, self.nt)?
                .f_sum(Kind::Int64)?
                .int64_value(&[]);
            if 4 * alive < 3 * self.nt {
                let (index, v, opp) = self.compact(self.cap)?;
                let _ = self.locals.keep("new_index", index.shallow_clone());
                self.v = v;
                self.opp = opp;
                self.pt.f_copy_(
                    &index
                        .f_index(&[Some(&self.pt.f_clamp_min(0)?)])?
                        .f_where_self(&self.pt.f_ge(0)?, &self.pt)?,
                )?;
                let _ = self.alive.f_zero_()?;
                let _ = self.alive.f_narrow(0, 0, alive)?.f_fill_(1)?;
                self.nt = alive;
            }
        }
        self.flip(crate::native::clone_tensor(&self.alive)?)?;
        Ok(())
    }
}
fn run(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let points = tensor(a, kw, 0, "points")?;
    super::mesh::shape("points", &points, &[None, Some(3)])?;
    let n = points.size()[0];
    let device = points.device();
    if n == 0 {
        return Err(w4a8::err(
            "ValueError",
            "delaunay3d needs at least one point",
        ));
    }
    let stream = stream(rt, &points)?;
    let verts = Tensor::f_empty([n + 4, 3], (Kind::Double, device))?;
    verts.f_narrow(0, 0, n)?.f_copy_(&points)?;
    let (lo, _) = verts.f_narrow(0, 0, n)?.f_min_dim(0, false)?;
    let (hi, _) = verts.f_narrow(0, 0, n)?.f_max_dim(0, false)?;
    let sizes = Tensor::f_stack(
        &[
            hi.f_sub(&lo)?.f_norm()?.f_div_scalar(2.)?,
            verts.f_narrow(0, 0, n)?.f_abs()?.f_max()?,
        ],
        0,
    )?
    .f_to_device(tch::Device::Cpu)?;
    let radius = sizes.double_value(&[0]);
    let magnitude = sizes.double_value(&[1]);
    drop(sizes);
    let mut generator = rt.native.generator(device, 0)?;
    // Python max retains its first NaN instead of applying f64::max's NaN rule.
    let mut maximum = radius;
    for value in [magnitude, 1e-12] {
        if value > maximum {
            maximum = value;
        }
    }
    let jog = 1e-10 * maximum;
    let mut view = verts.f_narrow(0, 0, n)?;
    let _ = view.f_add_(
        &generator
            .rand(&[n, 3], Kind::Double)?
            .f_mul_scalar(2.)?
            .f_sub_scalar(1.)?
            .f_mul_scalar(jog)?,
    )?;
    drop(view);
    let corners = Tensor::f_from_slice(&[1_f64, 1., 1., 1., -1., -1., -1., 1., -1., -1., -1., 1.])?
        .f_reshape([4, 3])?
        .f_to_device(device)?;
    let bound = if 10. * jog > radius {
        10. * jog
    } else {
        radius
    };
    verts.f_narrow(0, n, 4)?.f_copy_(
        &lo.f_add(&hi)?
            .f_div_scalar(2.)?
            .f_add(&corners.f_mul_scalar(20. * bound)?)?,
    )?;
    let cap = (8 * n + 64).min(MAX_TETS);
    let v = Tensor::f_full([cap, 4], -1, (Kind::Int, device))?;
    v.f_get(0)?.f_copy_(
        &Tensor::f_from_slice(&[n as i32, n as i32 + 1, n as i32 + 3, n as i32 + 2])?
            .f_to_device(device)?,
    )?;
    let opp = Tensor::f_full([cap, 4], -1, (Kind::Int, device))?;
    let alive = Tensor::f_zeros([cap], (Kind::Uint8, device))?;
    let _ = alive.f_get(0)?.f_fill_(1)?;
    let pend = Tensor::f_zeros([4 * cap], (Kind::Uint8, device))?;
    let pt = Tensor::f_zeros([n], (Kind::Int, device))?;
    let prio = generator.randperm(n)?.f_to_kind(Kind::Int)?;
    let to_pt = prio.f_argsort(-1, false)?.f_to_kind(Kind::Int)?;
    let mut state = State {
        rt,
        n,
        cap,
        nt: 1,
        stream,
        verts,
        v,
        opp,
        alive,
        pend,
        pt,
        prio,
        to_pt,
        locals: SourceLocals::default(),
    };
    // Source main-frame tensors remain live while nested grow/flip/compact run.
    let _ = state.locals.keep("lo", lo.shallow_clone());
    let _ = state.locals.keep("hi", hi.shallow_clone());
    let _ = state.locals.keep("corners", corners.shallow_clone());
    state.insert_all()?;
    let rows = state
        .alive
        .f_narrow(0, 0, state.nt)?
        .f_sum(Kind::Int64)?
        .int64_value(&[]);
    let (new_index, tets, neighbors) = state.compact(rows)?;
    let _ = state.locals.keep("_", new_index.shallow_clone());
    Ok(Value::Tuple(vec![
        Value::tensor(state.verts),
        Value::tensor(tets),
        Value::tensor(
            neighbors
                .f_bitwise_right_shift_tensor_scalar(2)?
                .f_where_self(&neighbors.f_ge(0)?, &neighbors)?,
        ),
    ]))
}
pub(super) fn install(rt: &Runtime) {
    rt.backends.bind_reference("cuda", "delaunay3d", body(run));
}
