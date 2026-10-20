//! Outer VAE policy bound to the same physical owner and memory coordinator as
//! the text encoder and DiT. Forward snapshots do not survive a load transition.
use super::*;
use crate::{
    execution::Runtime,
    memory::{Handle, Manager, MemoryServices},
    native::Error,
};
use std::rc::Rc;
use tch::{Device, Tensor};

pub struct ManagedImageVae<'a, 'b> {
    pub runtime: &'a Runtime,
    pub model: Rc<StaticModel<'a, ComponentBuffers>>,
    pub patcher: Handle<'a>,
    pub manager: &'b mut Manager<'a>,
    pub memory: &'b mut dyn MemoryServices,
    pub execution: &'b mut ResidentExecution,
    pub config: ImageVaeConfig,
    pub device: Device,
    pub force_full_load: bool,
}
impl ImageVaeServices for ManagedImageVae<'_, '_> {
    fn load(&mut self, memory_required: f64) -> Result<(), Error> {
        self.manager.load(
            std::slice::from_ref(&self.patcher),
            memory_required,
            None,
            false,
            self.force_full_load,
            self.memory,
        )
    }
    fn free_memory(&mut self) -> Result<f64, Error> {
        Ok(self.memory.free(self.device)?.0)
    }
    fn empty_cache(&mut self) -> Result<(), Error> {
        self.memory.empty_cache()
    }
    fn call(&mut self, operation: VaeOperation, input: &Tensor) -> Result<Tensor, Error> {
        let parameters = self.model.owner.borrow().execution_parameters();
        let vae = ImageVae::new(self.config.clone(), parameters);
        match operation {
            VaeOperation::Encode => vae.encode(self.runtime, input, self.execution),
            VaeOperation::Decode => vae.decode(self.runtime, input, self.execution),
        }
    }
    fn is_oom(&mut self, error: &Error) -> Result<bool, Error> {
        image_vae_is_oom(&self.runtime.native, self.device, error)
    }
}
