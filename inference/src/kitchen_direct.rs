//! Public Kitchen CUDA helpers that deliberately bypass the backend registry.
use crate::{
    execution::Runtime,
    native::{Error, KernelArg},
};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};
use tch::{Device, Kind, Tensor};
fn runtime(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "RuntimeError",
        message: message.into(),
    }
}
fn dims<const N: usize>(t: &Tensor) -> Result<[i64; N], Error> {
    let shape = t.size();
    shape
        .try_into()
        .map_err(|s: Vec<i64>| Error::ReferenceException {
            class: "ValueError",
            message: if s.len() < N {
                format!(
                    "not enough values to unpack (expected {N}, got {})",
                    s.len()
                )
            } else {
                format!("too many values to unpack (expected {N})")
            },
        })
}
fn dim(t: &Tensor, n: usize) -> Result<i64, Error> {
    t.size().get(n).copied().ok_or(Error::ReferenceException {
        class: "IndexError",
        message: "tuple index out of range".into(),
    })
}
fn cuda_device(rt: &Runtime, device: Option<Device>) -> Result<Option<usize>, Error> {
    if device.is_some_and(|d| !matches!(d, Device::Cuda(_))) || !rt.native.cuda_available()? {
        return Ok(None);
    }
    Ok(Some(match device {
        Some(Device::Cuda(i)) => i,
        _ => rt.native.current_device()?,
    }))
}
pub fn flash_num_splits(
    batch_heads: i64,
    capacity: i64,
    multiprocessors: i64,
    block_n: i64,
) -> Result<i64, Error> {
    if block_n == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "integer division or modulo by zero".into(),
        });
    }
    let blocks = (capacity + block_n - 1).div_euclid(block_n);
    let max_splits = 32.min(multiprocessors * 2).min(blocks);
    let mut efficiencies = vec![];
    let mut best = 0.0f64;
    for splits in 1..=max_splits {
        let eligible = splits == 1
            || (blocks as f64 / splits as f64).ceil()
                != (blocks as f64 / (splits - 1) as f64).ceil();
        let efficiency = batch_heads as f64 * splits as f64 / (multiprocessors * 2) as f64;
        if eligible && efficiency.ceil() == 0. {
            return Err(Error::ReferenceException {
                class: "ZeroDivisionError",
                message: "float division by zero".into(),
            });
        }
        let efficiency = if eligible {
            efficiency / efficiency.ceil()
        } else {
            0.
        };
        efficiencies.push(efficiency);
        best = best.max(efficiency);
    }
    efficiencies
        .iter()
        .position(|e| *e >= 0.85 * best)
        .map(|i| i as i64 + 1)
        .ok_or(Error::ReferenceException {
            class: "StopIteration",
            message: String::new(),
        })
}
#[derive(Clone, Copy)]
pub struct DeltaArgs<'a> {
    pub mixed_qkv: &'a Tensor,
    pub x: &'a Tensor,
    pub w_a: &'a Tensor,
    pub w_b: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub g_decay: &'a Tensor,
    pub state: &'a Tensor,
    pub key_dim: i64,
    pub num_key_heads: i64,
    pub scale: f64,
    pub z: &'a Tensor,
    pub norm_weight: &'a Tensor,
    pub eps: f64,
    pub snapshots: Option<&'a Tensor>,
}
impl Runtime {
    pub fn flash_decode_available(&self, device: Option<Device>) -> Result<bool, Error> {
        let Some(i) = cuda_device(self, device)? else {
            return Ok(false);
        };
        let props = self.native.device_properties(i)?;
        Ok((props.major, props.minor) >= (8, 0))
    }
    pub fn gated_delta_available(
        &self,
        device: Option<Device>,
        key_head_dim: i64,
        value_head_dim: i64,
    ) -> Result<bool, Error> {
        let Some(i) = cuda_device(self, device)? else {
            return Ok(false);
        };
        if key_head_dim != 128 || value_head_dim % 32 != 0 || !(1..=512).contains(&value_head_dim) {
            return Ok(false);
        }
        let props = self.native.device_properties(i)?;
        if (props.major, props.minor) < (7, 5) {
            return Ok(false);
        }
        static OPTIN: OnceLock<Mutex<BTreeMap<usize, i64>>> = OnceLock::new();
        let mut cache = OPTIN.get_or_init(Default::default).lock().unwrap();
        let optin = *cache
            .entry(i)
            .or_insert(props.shared_memory_per_block_optin);
        let warps = value_head_dim.max(128) / 32;
        Ok((key_head_dim * value_head_dim + 2 * 8 * key_head_dim + 4 * 8 * warps) * 4 <= optin)
    }
    /// # Safety
    /// Same original packed CUDA argument capacities and stream contract.
    pub unsafe fn flash_attention_decode(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        lengths: &Tensor,
    ) -> Result<Tensor, Error> {
        let [batch, _, query_heads, head_dim] = dims(q)?;
        let [_, capacity, kv_heads, _] = dims(k)?;
        if !self.flash_decode_available(Some(q.device()))? {
            return Err(runtime(
                "flash_attention_decode requires the CUDA extension on SM80 or newer",
            ));
        }
        if kv_heads == 0 {
            return Err(Error::ReferenceException {
                class: "ZeroDivisionError",
                message: "integer division or modulo by zero".into(),
            });
        }
        let groups = query_heads.div_euclid(kv_heads);
        let query = q
            .f_reshape([batch, kv_heads, groups, head_dim])?
            .f_transpose(1, 2)?
            .f_reshape([batch * groups, kv_heads, head_dim])?;
        let out = Tensor::f_empty_like(&query)?;
        let Device::Cuda(i) = q.device() else {
            unreachable!()
        };
        let splits = flash_num_splits(
            batch * kv_heads,
            capacity,
            self.native.device_properties(i)?.multiprocessor_count,
            if head_dim == 256 { 64 } else { 128 },
        )?;
        let lse = Tensor::f_empty([batch * kv_heads * groups], (Kind::Float, q.device()))?;
        let (lse_acc, out_acc) = if splits > 1 {
            (
                Tensor::f_empty([splits, batch, kv_heads, groups], (Kind::Float, q.device()))?,
                Tensor::f_empty(
                    [splits, batch, kv_heads, groups, head_dim],
                    (Kind::Float, q.device()),
                )?,
            )
        } else {
            let zero = lse.f_slice(0, 0, 0, 1)?;
            (zero.shallow_clone(), zero)
        };
        use KernelArg::{Int, Tensor as T};
        unsafe {
            self.native.call_unchecked(
                "flash_attention_decode",
                &[
                    T(&query),
                    T(k),
                    T(v),
                    T(lengths),
                    T(&out),
                    T(&lse),
                    T(&lse_acc),
                    T(&out_acc),
                    Int(splits),
                    Int(self.native.current_stream(i)? as i64),
                ],
            )?;
        }
        Ok(out
            .f_view([batch, groups, kv_heads, head_dim])?
            .f_transpose(1, 2)?
            .f_reshape_as(q)?)
    }
    /// # Safety
    /// Same original state/snapshot capacities, dtype and stream contracts.
    pub unsafe fn gated_delta_decode_fused(&self, args: DeltaArgs<'_>) -> Result<Tensor, Error> {
        let a = args;
        let [batch, _, seq] = dims(a.mixed_qkv)?;
        let heads = dim(a.state, 1)?;
        let dk = dim(a.state, 2)?;
        let dv = dim(a.state, 3)?;
        if !self.gated_delta_available(Some(a.x.device()), dk, dv)? {
            return Err(runtime(
                "gated_delta_decode_fused is unavailable for this device and head shape",
            ));
        }
        let out = Tensor::f_empty([batch, seq, heads, dv], (a.x.kind(), a.x.device()))?;
        let qkv = a.mixed_qkv.f_contiguous()?;
        let x = a.x.f_contiguous()?;
        let wa = a.w_a.f_contiguous()?;
        let wb = a.w_b.f_contiguous()?;
        let z = a.z.f_reshape([batch, seq, heads * dv])?.f_contiguous()?;
        let nw = a.norm_weight.f_contiguous()?;
        let Device::Cuda(i) = x.device() else {
            unreachable!()
        };
        use KernelArg::{Float, Int, Tensor as T};
        let result = unsafe {
            self.native.call_unchecked(
                "gated_delta_decode_fused",
                &[
                    T(&qkv),
                    T(&x),
                    T(&wa),
                    T(&wb),
                    T(a.dt_bias),
                    T(a.g_decay),
                    T(a.state),
                    T(&out),
                    a.snapshots.map_or(KernelArg::None, T),
                    T(&z),
                    T(&nw),
                    Float(a.eps),
                    Int(a.key_dim),
                    Int(a.num_key_heads),
                    Float(a.scale),
                    Int(self.native.current_stream(i)? as i64),
                ],
            )?
        };
        if result.value == 0 {
            return Err(runtime("gated_delta_decode_fused launch rejected"));
        }
        Ok(out)
    }
    /// # Safety
    /// Same original state/snapshot capacities, dtype and stream contracts.
    pub unsafe fn deltanet_conv_step(
        &self,
        proj: &Tensor,
        state: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        snapshots: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        if !self.gated_delta_available(Some(proj.device()), 128, 128)? {
            return Err(runtime(
                "deltanet_conv_step requires the CUDA or HIP extension",
            ));
        }
        let [batch, seq, channels] = dims(proj)?;
        let out = Tensor::f_empty([batch, channels, seq], (proj.kind(), proj.device()))?;
        let proj = proj.f_contiguous()?;
        let weight = weight.f_reshape([channels, -1])?.f_contiguous()?;
        let bias = bias.map(Tensor::f_contiguous).transpose()?;
        let Device::Cuda(i) = proj.device() else {
            unreachable!()
        };
        use KernelArg::{Int, Tensor as T};
        let result = unsafe {
            self.native.call_unchecked(
                "deltanet_conv_step",
                &[
                    T(&proj),
                    T(state),
                    T(&weight),
                    bias.as_ref().map_or(KernelArg::None, T),
                    T(&out),
                    snapshots.map_or(KernelArg::None, T),
                    Int(self.native.current_stream(i)? as i64),
                ],
            )?
        };
        if result.value == 0 {
            return Err(runtime("deltanet_conv_step launch rejected"));
        }
        Ok(out)
    }
}
