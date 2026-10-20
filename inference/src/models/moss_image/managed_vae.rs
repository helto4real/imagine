//! Flux.2 VAE bound to the same original static owner and memory coordinator.
use super::{Flux2Vae, Flux2VaeConfig, MossStaticExecution};
use crate::{
    execution::Runtime,
    memory::{Handle, Manager, MemoryServices},
    models::qwen_image21::{
        ComponentBuffers, ImageVaeServices, StaticModel, VaeOperation, image_vae_is_oom,
    },
    native::Error,
};
use std::rc::Rc;
use tch::{Device, Tensor};
pub struct ManagedFlux2Vae<'a, 'b> {
    pub runtime: &'a Runtime,
    pub model: Rc<StaticModel<'a, ComponentBuffers>>,
    pub patcher: Handle<'a>,
    pub manager: &'b mut Manager<'a>,
    pub memory: &'b mut dyn MemoryServices,
    pub execution: &'b mut MossStaticExecution,
    pub config: Flux2VaeConfig,
    pub device: Device,
    pub force_full_load: bool,
}
impl ImageVaeServices for ManagedFlux2Vae<'_, '_> {
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
        let vae = Flux2Vae::new(self.config.clone(), parameters);
        match operation {
            VaeOperation::Encode => vae.encode(self.runtime, input, self.execution),
            VaeOperation::Decode => vae.decode(self.runtime, input, self.execution),
        }
    }
    fn is_oom(&mut self, error: &Error) -> Result<bool, Error> {
        image_vae_is_oom(&self.runtime.native, self.device, error)
    }
}
