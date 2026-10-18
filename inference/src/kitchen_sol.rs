//! Original chunked-producer Sol attention and persistent RoPE allocation context.
use crate::{execution::Runtime, execution_value::Value, native::Error};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use tch::{Device, Kind, Tensor};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
/// The original process-wide allocation context. It is entered only where the source does.
pub trait AllocationContext: Send + Sync {
    fn enter(&self) -> Result<(), Error>;
    fn exit(&self, failure: Option<&Error>) -> Result<bool, Error>;
}
fn context() -> &'static RwLock<Option<Arc<dyn AllocationContext>>> {
    static CONTEXT: OnceLock<RwLock<Option<Arc<dyn AllocationContext>>>> = OnceLock::new();
    CONTEXT.get_or_init(Default::default)
}
pub fn set_allocation_context(value: Option<Arc<dyn AllocationContext>>) {
    *context().write().unwrap() = value;
}
pub fn allocation_context() -> Option<Arc<dyn AllocationContext>> {
    context().read().unwrap().clone()
}
enum TensorWeak {
    Plain(Weak<Mutex<crate::execution_value::TensorObject>>),
    Quantized(Weak<Mutex<crate::storage::QuantizedStorage>>),
}
impl TensorWeak {
    fn new(value: &Value) -> Self {
        match value {
            Value::Tensor(t) => Self::Plain(Arc::downgrade(t)),
            Value::Quantized(q) => Self::Quantized(Arc::downgrade(q)),
            _ => unreachable!(),
        }
    }
    fn same(&self, value: &Value) -> bool {
        match (self, value) {
            (Self::Plain(w), Value::Tensor(t)) => w.upgrade().is_some_and(|v| Arc::ptr_eq(&v, t)),
            (Self::Quantized(w), Value::Quantized(q)) => {
                w.upgrade().is_some_and(|v| Arc::ptr_eq(&v, q))
            }
            _ => false,
        }
    }
}
struct RopeFab {
    source: TensorWeak,
    t: i64,
    rot: i64,
    tensor: Tensor,
}
fn cache() -> &'static Mutex<Option<RopeFab>> {
    static CACHE: OnceLock<Mutex<Option<RopeFab>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}
fn metadata(value: &Value) -> Result<(Vec<i64>, Device), Error> {
    match value {
        Value::Tensor(t) => {
            let t = t.lock().unwrap();
            Ok((t.size(), t.device()))
        }
        Value::Quantized(q) => {
            let q = q.lock().unwrap();
            Ok((q.params.original_shape.clone(), q.payload.device()))
        }
        _ => {
            let name = match value {
                Value::None => "NoneType",
                Value::Bool(_) => "bool",
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::Text(_) => "str",
                Value::DType(_) => "torch.dtype",
                Value::Device(_) => "torch.device",
                Value::Tuple(_) => "tuple",
                Value::List(_) => "list",
                Value::Dict(_) => "dict",
                _ => unreachable!(),
            };
            Err(error(
                "AttributeError",
                format!("'{name}' object has no attribute 'shape'"),
            ))
        }
    }
}
fn allocate(shape: [i64; 3], device: Device) -> Result<Tensor, Error> {
    let ctx = context().read().unwrap().clone();
    if let Some(ctx) = &ctx {
        ctx.enter()?;
    }
    let allocated = Tensor::f_empty(shape, (Kind::Float, device)).map_err(Error::from);
    let suppressed = if let Some(ctx) = ctx {
        ctx.exit(allocated.as_ref().err())?
    } else {
        false
    };
    if allocated.is_err() && suppressed {
        return Err(error(
            "UnboundLocalError",
            "cannot access local variable 'fab' where it is not associated with a value",
        ));
    }
    allocated
}
impl Runtime {
    /// # Safety
    /// A quantized input's original decoder must receive supported operands.
    pub unsafe fn sol_packed_rope_fab(&self, freqs: &Value, t: i64) -> Result<Tensor, Error> {
        let (s, device) = metadata(freqs)?;
        let rot = s
            .get(s.len().wrapping_sub(3))
            .copied()
            .ok_or_else(|| error("IndexError", "tuple index out of range"))?
            * 2;
        if rot % 8 != 0 || rot <= 0 || rot > 128 {
            return Err(error(
                "ValueError",
                format!("sol_attn_chunked: rot_dim must be a multiple of 8 in (0, 128], got {rot}"),
            ));
        }
        {
            let cached = cache().lock().unwrap();
            if let Some(c) = &*cached
                && c.t == t
                && c.rot == rot
                && c.source.same(freqs)
            {
                return Ok(c.tensor.shallow_clone());
            }
        }
        let f = unsafe {
            self.op(
                "aten::reshape.default",
                &[
                    freqs.clone(),
                    Value::List(vec![
                        Value::Int(-1),
                        Value::Int(rot / 2),
                        Value::Int(2),
                        Value::Int(2),
                    ]),
                ],
                &vec![],
            )?
        };
        let (actual, _) = metadata(&f)?;
        if actual[0] != t {
            return Err(error(
                "ValueError",
                format!("sol_attn: rope_freqs covers {} tokens, T={t}", actual[0]),
            ));
        }
        let fab = allocate([t, rot, 2], device)?;
        let select = |a: i64, b: i64| -> Result<Tensor, Error> {
            let first = unsafe {
                self.op(
                    "aten::select.int",
                    &[f.clone(), Value::Int(2), Value::Int(a)],
                    &vec![],
                )?
            };
            unsafe {
                self.op(
                    "aten::select.int",
                    &[first, Value::Int(2), Value::Int(b)],
                    &vec![],
                )?
            }
            .as_tensor()
        };
        fab.f_slice(1, 0, rot / 2, 1)?
            .f_select(2, 0)?
            .f_copy_(&select(0, 0)?)?;
        fab.f_slice(1, 0, rot / 2, 1)?
            .f_select(2, 1)?
            .f_copy_(&select(0, 1)?)?;
        fab.f_slice(1, rot / 2, rot, 1)?
            .f_select(2, 0)?
            .f_copy_(&select(1, 1)?)?;
        fab.f_slice(1, rot / 2, rot, 1)?
            .f_select(2, 1)?
            .f_copy_(&select(1, 0)?)?;
        *cache().lock().unwrap() = Some(RopeFab {
            source: TensorWeak::new(freqs),
            t,
            rot,
            tensor: fab.shallow_clone(),
        });
        Ok(fab)
    }
}

use crate::execution_bindings::backends::sol_cuda as shared;
use crate::{
    native::KernelArg,
    registry::{Argument, CallArgs},
};
use std::collections::BTreeMap;
pub type ChunkIterator<'a> = Box<dyn Iterator<Item = Result<Tensor, Error>> + 'a>;
pub enum ChunkSource<'a> {
    Iterator(Option<ChunkIterator<'a>>),
    Replay(Vec<Tensor>),
    Factory(Box<dyn FnMut() -> Result<ChunkIterator<'a>, Error> + 'a>),
}
impl ChunkSource<'_> {
    fn pass(&mut self) -> Result<ChunkIterator<'_>, Error> {
        match self {
            Self::Iterator(iterator) => Ok(iterator
                .take()
                .unwrap_or_else(|| Box::new(std::iter::empty()))),
            Self::Replay(chunks) => Ok(Box::new(chunks.iter().map(|t| Ok(t.shallow_clone())))),
            Self::Factory(factory) => factory(),
        }
    }
    fn materialize_for_bootstrap(&mut self) -> Result<(), Error> {
        if let Self::Iterator(iterator) = self {
            let chunks = iterator
                .take()
                .unwrap_or_else(|| Box::new(std::iter::empty()))
                .collect::<Result<Vec<_>, _>>()?;
            *self = Self::Replay(chunks);
        }
        Ok(())
    }
}
pub struct SolChunkOptions<'a> {
    pub kmean: Option<&'a Tensor>,
    pub vscale: Option<&'a Tensor>,
    pub tau: f64,
    pub topk_ratio: f64,
    pub scale: Option<f64>,
    pub sink_blocks: Option<[i64; 2]>,
    pub sink_q: Option<[i64; 2]>,
    pub rope_eps: f64,
    pub tail: bool,
    pub block_len: Option<&'a Tensor>,
    pub coarse_gate: Option<&'a Tensor>,
    pub token_aug: i64,
}
impl Default for SolChunkOptions<'_> {
    fn default() -> Self {
        Self {
            kmean: None,
            vscale: None,
            tau: 1.,
            topk_ratio: 0.,
            scale: None,
            sink_blocks: None,
            sink_q: None,
            rope_eps: 1e-6,
            tail: true,
            block_len: None,
            coarse_gate: None,
            token_aug: 0,
        }
    }
}
fn tuple(s: &[i64]) -> String {
    format!(
        "({}{})",
        s.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
        if s.len() == 1 { "," } else { "" }
    )
}
fn device_name(d: Device) -> String {
    match d {
        Device::Cuda(i) => format!("cuda:{i}"),
        _ => crate::execution_value::device_name(d),
    }
}
fn block_lengths_input(
    block: Option<&Tensor>,
    t: i64,
    device: Device,
) -> Result<Option<Tensor>, Error> {
    let Some(block) = block else { return Ok(None) };
    let n = (t + 63) / 64;
    if block.kind() != Kind::Int || block.dim() != 1 || block.numel() as i64 != n {
        return Err(error(
            "ValueError",
            format!(
                "sol_attn: block_len must be int32 of shape ({n},), got {} {}",
                crate::registry::AllowedType::DType(block.kind()).python_name(),
                tuple(&block.size())
            ),
        ));
    }
    if block.device() != device {
        return Err(error(
            "ValueError",
            format!(
                "sol_attn: block_len must be on {}, got {}",
                device_name(device),
                device_name(block.device())
            ),
        ));
    }
    Ok(Some(block.f_contiguous()?))
}
fn ksums(ws: &Tensor, p: &BTreeMap<String, i64>, h: i64) -> Result<Tensor, Error> {
    Ok(ws
        .f_narrow(0, p["scratch"], h * p["NPAD"] * 128 * 4)?
        .f_view_dtype(Kind::Float)?
        .f_view([h, p["NPAD"], 128])?
        .f_slice(1, 0, p["NTB"], 1)?)
}
fn value_scale(x: &Tensor) -> Result<Tensor, Error> {
    Ok(x.f_div_scalar(127.)?.f_mul_scalar(1.1)?.f_clamp_min(1e-8)?)
}
struct Produce<'a> {
    rt: &'a Runtime,
    ws: &'a Tensor,
    fab: &'a Tensor,
    qw: &'a Tensor,
    kw: &'a Tensor,
    t: i64,
    h: i64,
    rot: i64,
    block: Option<&'a Tensor>,
    options: &'a SolChunkOptions<'a>,
    stream: i64,
}
impl Produce<'_> {
    unsafe fn run(
        &self,
        source: &mut ChunkSource<'_>,
        kmean: &Tensor,
        vscale: &Tensor,
    ) -> Result<(), Error> {
        use KernelArg::{Float, Int, Tensor as T};
        unsafe {
            self.rt.native.call_unchecked(
                "sol_producer_begin",
                &[
                    T(self.ws),
                    Int(1),
                    Int(self.t),
                    Int(self.h),
                    Int(self.stream),
                    Int(self.options.token_aug),
                ],
            )?;
        }
        let mut t0 = 0;
        for chunk in source.pass()? {
            let chunk = chunk?;
            let s = chunk.size();
            let m = *s
                .get(s.len().wrapping_sub(2))
                .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
            if t0 % 64 != 0 && m != 0 {
                return Err(error(
                    "ValueError",
                    "sol_attn_chunked: chunk starts must be 64-aligned",
                ));
            }
            if s[s.len() - 1] != 3 * self.h * 128
                || chunk.kind() != Kind::BFloat16
                || chunk.device() != self.fab.device()
            {
                return Err(error(
                    "ValueError",
                    format!(
                        "sol_attn_chunked: chunks must be [M, {}] bfloat16 on {}, got {} {} on {}",
                        3 * self.h * 128,
                        device_name(self.fab.device()),
                        tuple(&s),
                        crate::registry::AllowedType::DType(chunk.kind()).python_name(),
                        device_name(chunk.device())
                    ),
                ));
            }
            let chunk = chunk.f_contiguous()?;
            unsafe {
                self.rt.native.call_unchecked(
                    "sol_producer_chunk",
                    &[
                        T(self.ws),
                        T(&chunk),
                        T(self.fab),
                        T(self.qw),
                        T(self.kw),
                        T(kmean),
                        T(vscale),
                        Float(self.options.rope_eps),
                        Int(self.rot),
                        Int(t0),
                        Int(m),
                        Int(1),
                        Int(self.t),
                        Int(self.h),
                        Int(self.stream),
                        self.block.map_or(KernelArg::None, T),
                        Int(self.options.token_aug),
                    ],
                )?;
            }
            t0 += m;
        }
        if t0 != self.t {
            return Err(error(
                "ValueError",
                format!("sol_attn_chunked: chunks cover {t0} tokens, T={}", self.t),
            ));
        }
        Ok(())
    }
}
impl Runtime {
    /// # Safety
    /// Chunks and statistics must meet the frozen native pointer capacities and stream contract.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sol_attn_chunked(
        &self,
        mut source: ChunkSource<'_>,
        t: i64,
        h: i64,
        freqs: &Value,
        norm_weights: (&Tensor, &Tensor),
        options: &SolChunkOptions<'_>,
    ) -> Result<(Tensor, Tensor, Tensor), Error> {
        let fab = unsafe { self.sol_packed_rope_fab(freqs, t)? };
        let rot = fab.size()[1];
        let dev = fab.device();
        let Device::Cuda(device) = dev else {
            return Err(error(
                "ValueError",
                format!("Expected a cuda device, but got: {}", device_name(dev)),
            ));
        };
        let prop = self.native.device_properties(device)?;
        if prop.major < 8 {
            return Err(error(
                "RuntimeError",
                format!(
                    "sol_attn: requires sm_80+ (cp.async, INT8 MMA); {} is sm_{}{}",
                    device_name(dev),
                    prop.major,
                    prop.minor
                ),
            ));
        }
        let mut args = CallArgs::new();
        args.insert(
            "sink_blocks".into(),
            options
                .sink_blocks
                .map_or(Argument::None, |v| Argument::Sequence(v.to_vec())),
        );
        args.insert(
            "sink_q".into(),
            options
                .sink_q
                .map_or(Argument::None, |v| Argument::Sequence(v.to_vec())),
        );
        args.insert(
            "topk_ratio".into(),
            Argument::FloatValue(options.topk_ratio),
        );
        if let Err(f) = crate::backend_contract::validate_sol(&args) {
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
        let scale = options.scale.unwrap_or(128f64.powf(-0.5));
        let block = block_lengths_input(options.block_len, t, dev)?;
        if let Some(gate) = options.coarse_gate
            && (gate.size() != [1, t, h, 128] || gate.device() != dev)
        {
            return Err(error(
                "ValueError",
                format!(
                    "sol_attn: coarse_gate must be {} on {}, got {} on {}",
                    tuple(&[1, t, h, 128]),
                    device_name(dev),
                    tuple(&gate.size()),
                    device_name(gate.device())
                ),
            ));
        }
        let lengths = shared::lengths(t, (t + 63) / 64, dev, block.as_ref())?;
        let qw = norm_weights
            .0
            .f_to_device_(dev, Kind::BFloat16, false, false)?
            .f_contiguous()?;
        let kw = norm_weights
            .1
            .f_to_device_(dev, Kind::BFloat16, false, false)?
            .f_contiguous()?;
        let bootstrap = options.kmean.is_none() || options.vscale.is_none();
        if bootstrap {
            source.materialize_for_bootstrap()?;
        }
        use KernelArg::{Bool, Float, Int, Tensor as T};
        let plan = unsafe {
            self.native.call_unchecked(
                "sol_attn_plan",
                &[Int(1), Int(t), Int(h), Int(options.token_aug)],
            )?
        };
        let p: BTreeMap<String, i64> = serde_json::from_str(
            plan.metadata
                .as_deref()
                .ok_or_else(|| Error::Native("Missing native Sol plan".into()))?,
        )
        .map_err(|e| Error::Native(e.to_string()))?;
        let ws = Tensor::f_empty([p["total"]], (Kind::Uint8, dev))?;
        let stream = self.native.current_stream(device)? as i64;
        let producer = Produce {
            rt: self,
            ws: &ws,
            fab: &fab,
            qw: &qw,
            kw: &kw,
            t,
            h,
            rot,
            block: block.as_ref(),
            options,
            stream,
        };
        let (kmean, vscale) = if bootstrap {
            let zero = Tensor::f_zeros([h, 128], (self.native.default_dtype()?, dev))?;
            let one = Tensor::f_ones([h, 128], (self.native.default_dtype()?, dev))?;
            unsafe {
                producer.run(&mut source, &zero, &one)?;
            }
            let km = ksums(&ws, &p, h)?
                .f_sum_dim_intlist([1].as_slice(), false, None)?
                .f_div(&lengths.f_sum(None)?)?;
            let vs = value_scale(
                &ws.f_narrow(0, p["statsV"], h * 128 * 4)?
                    .f_view_dtype(Kind::Float)?
                    .f_view([h, 128])?,
            )?;
            (km, vs)
        } else {
            (
                options.kmean.unwrap().shallow_clone(),
                options.vscale.unwrap().shallow_clone(),
            )
        };
        let kmean = kmean
            .f_to_device_(dev, Kind::Float, false, false)?
            .f_contiguous()?;
        let vscale = vscale
            .f_to_device_(dev, Kind::Float, false, false)?
            .f_clamp_min(1e-8)?
            .f_contiguous()?;
        unsafe {
            producer.run(&mut source, &kmean, &vscale)?;
        }
        let sinks = options.sink_blocks.unwrap_or([0, 0]);
        let sinkq = options.sink_q.unwrap_or([0, 0]);
        let threshold = if options.topk_ratio != 0. {
            let n = p["NTB"];
            let cen8 = ws
                .f_narrow(0, p["cen8"], h * n * 128)?
                .f_view_dtype(Kind::Int8)?
                .f_view([h, n, 128])?
                .f_to_kind(Kind::Float)?;
            let cens = ws
                .f_narrow(0, p["cens"], h * n * 4)?
                .f_view_dtype(Kind::Float)?
                .f_view([h, n])?;
            let kc = ksums(&ws, &p, h)?.f_div(&lengths.f_view([1, -1, 1])?)?;
            let kc = kc.f_sub(&kc.f_mean_dim([1].as_slice(), true, None)?)?;
            let d = Tensor::f_arange(128, (Kind::Int64, Device::Cpu))?;
            let channel_block = d.f_bitwise_right_shift_tensor_scalar(5)?;
            let rem = d.f_bitwise_and(31)?;
            let hh = rem.f_bitwise_right_shift_tensor_scalar(4)?;
            let r2 = rem.f_bitwise_and(15)?;
            let perm = channel_block
                .f_mul_scalar(32)?
                .f_add(&r2.f_bitwise_right_shift_tensor_scalar(2)?.f_mul_scalar(8)?)?
                .f_add(&hh.f_mul_scalar(4)?)?
                .f_add(&r2.f_bitwise_and(3)?)?
                .f_to_device(dev)?;
            let cen8 = cen8.f_index_select(2, &perm)?;
            Some(shared::topk_from_pooled(
                &cen8,
                &cens,
                &kc,
                options.topk_ratio,
                scale * std::f64::consts::LOG2_E,
                (sinks[0], sinks[1]),
            )?)
        } else {
            None
        };
        let mut out = Tensor::f_empty([1, t, h, 128], (Kind::BFloat16, dev))?;
        let knext = Tensor::f_empty([h, 128], (Kind::Float, dev))?;
        let vamax = Tensor::f_empty([h, 128], (Kind::Float, dev))?;
        unsafe {
            self.native.call_unchecked(
                "sol_attn_core",
                &[
                    T(&ws),
                    T(&out),
                    T(&vscale),
                    T(&knext),
                    T(&vamax),
                    Int(1),
                    Int(t),
                    Int(h),
                    Float(options.tau),
                    Float(scale),
                    Int(sinks[0]),
                    Int(sinks[1]),
                    Int(sinkq[0]),
                    Int(sinkq[1]),
                    Int(stream),
                    threshold.as_ref().map_or(KernelArg::None, T),
                    block.as_ref().map_or(KernelArg::None, T),
                    Bool(options.tail),
                    Int(options.token_aug),
                ],
            )?;
        }
        if let Some(gate) = options.coarse_gate {
            shared::coarse(&mut out, &ws, &p, &lengths, gate, scale)?;
        }
        Ok((out, knext, value_scale(&vamax)?))
    }
}
