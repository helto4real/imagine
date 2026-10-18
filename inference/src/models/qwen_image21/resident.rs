//! Explicit static/resident smoke-driver services. These are selected test
//! settings, never substitutes for dynamic vbars, host staging or malloc graphs.
use super::*;
use crate::{
    attention::{Attention as SharedAttention, AttentionInput},
    checkpoint::PreparedParameters,
    execution::Runtime,
    native::{Error, Kitchen},
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tch::{Device, Kind, Tensor};
fn gap(s: impl Into<String>) -> Error {
    Error::BindingGap {
        backend: "static_resident_test".into(),
        operation: s.into(),
    }
}
pub struct ResidentExecution {
    pub tensor_observer: Option<TensorObserver>,
    pub cancelled: Arc<AtomicBool>,
    pub operations: usize,
    pub lifecycle: Vec<Lifecycle>,
    attention: SharedAttention,
    blocks: BTreeMap<usize, SharedAttention>,
}
pub type TensorObserver = Box<dyn FnMut(&str, &Tensor) -> Result<(), Error>>;
impl ResidentExecution {
    /// Caller must explicitly select static model patchers, compiler disabled,
    /// zero asynchronous offload streams, resident weights and unpatched models.
    pub fn bind(
        rt: &Runtime,
        state: &PreparedParameters,
        layers: usize,
        attention: SharedAttention,
    ) -> Result<Self, Error> {
        let mut blocks = BTreeMap::new();
        for i in 0..layers {
            let key = format!("transformer_blocks.{i}.attn.comfy_attention.config");
            let config = state
                .get(&key)
                .map(|v| -> Result<serde_json::Value, Error> {
                    let t = v.as_tensor()?.f_contiguous()?.f_view_dtype(Kind::Uint8)?;
                    let mut data = vec![0; t.numel()];
                    let len = data.len();
                    t.f_copy_data_u8(&mut data, len)?;
                    serde_json::from_slice(&data).map_err(|e| Error::Native(e.to_string()))
                })
                .transpose()?;
            let mut bound = attention.clone();
            bound.method = attention.preferred(rt, config.as_ref())?;
            blocks.insert(i, bound);
        }
        Ok(Self {
            tensor_observer: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            operations: 0,
            lifecycle: vec![],
            attention,
            blocks,
        })
    }
    fn interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        if self.cancelled.load(Ordering::Relaxed) {
            Err(Error::ReferenceException {
                class: "InterruptProcessingException",
                message: String::new(),
            })
        } else {
            Ok(())
        }
    }
}
impl DiTExecution for ResidentExecution {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.interrupt()
    }
    fn lifecycle(&mut self, event: Lifecycle) -> Result<(), Error> {
        self.lifecycle.push(event);
        Ok(())
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: i64,
        mask: Option<&Tensor>,
        block: usize,
        _preferred: Option<&serde_json::Value>,
    ) -> Result<Tensor, Error> {
        let selected = self
            .blocks
            .get(&block)
            .ok_or_else(|| gap(format!("Unbound attention block {block}")))?;
        unsafe {
            selected.run(
                rt,
                AttentionInput {
                    q,
                    k,
                    v,
                    heads,
                    mask,
                    skip_reshape: false,
                    skip_output_reshape: false,
                    enable_gqa: false,
                    scale: None,
                },
            )
        }
    }
}
impl TextEncoderExecution for ResidentExecution {
    fn observe(&mut self, name: &str, tensor: &Tensor) -> Result<(), Error> {
        if let Some(observer) = &mut self.tensor_observer {
            observer(name, tensor)?;
        }
        Ok(())
    }
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.interrupt()
    }
    fn queue_begin(&mut self, _layers: usize, _device: Device) -> Result<(), Error> {
        Ok(())
    }
    fn queue_end(&mut self, _device: Device) -> Result<(), Error> {
        Ok(())
    }
    fn block(
        &mut self,
        _index: usize,
        core: &mut dyn FnMut(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        core(self)
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        heads: i64,
        gqa: bool,
    ) -> Result<Tensor, Error> {
        unsafe {
            self.attention.run(
                rt,
                AttentionInput {
                    q,
                    k,
                    v,
                    heads,
                    mask,
                    skip_reshape: true,
                    skip_output_reshape: false,
                    enable_gqa: gqa,
                    scale: None,
                },
            )
        }
    }
}
impl VaeExecution for ResidentExecution {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.interrupt()
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor, Error> {
        vae_pytorch_attention(rt, q, k, v)
    }
}
pub struct ResidentCacheServices {
    pub native: Kitchen,
}
struct ResidentPending(Tensor);
impl PendingCacheTensor for ResidentPending {
    fn synchronize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn wait(self: Box<Self>, _device: Device) -> Result<Tensor, Error> {
        Ok(self.0)
    }
}
impl CacheServices for ResidentCacheServices {
    fn patcher_free_memory(&mut self, device: Device) -> Result<f64, Error> {
        self.free_memory(device)
    }
    fn free_memory(&mut self, device: Device) -> Result<f64, Error> {
        let Device::Cuda(i) = device else {
            return Err(gap("CPU system/cgroup memory service"));
        };
        let [free, _total, active, reserved] = self.native.cuda_memory(i)?;
        // Static-only isolated process has no AIMDO vbar allocations to reclaim.
        Ok((free + reserved - active) as f64)
    }
    fn ensure_pin_budget(&mut self, _bytes: usize, _evict_active: bool) -> Result<bool, Error> {
        Err(gap("Host cache pin-budget service"))
    }
    fn pin(&mut self, t: &Tensor, _evict_active: bool) -> Result<bool, Error> {
        if t.device() != Device::Cpu {
            Ok(false)
        } else {
            Err(gap("Host cache pin registration"))
        }
    }
    fn unpin(&mut self, _t: &Tensor) -> Result<(), Error> {
        Err(gap("Host cache unpin registration"))
    }
    fn prefetch(
        &mut self,
        t: &Tensor,
        _block: usize,
        kind: Option<Kind>,
        device: Device,
    ) -> Result<Box<dyn PendingCacheTensor>, Error> {
        if t.device() != device {
            return Err(gap("Host/GPU cache staging buffers"));
        }
        Ok(Box::new(ResidentPending(t.f_to_device_(
            device,
            kind.unwrap_or(t.kind()),
            true,
            false,
        )?)))
    }
    fn clear_staging(&mut self) -> Result<(), Error> {
        Ok(())
    }
}
