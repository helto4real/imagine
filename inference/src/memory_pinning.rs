//! Original host-registration bodies for native-sized tensor pointers/byte
//! counts. Registrations retain sizes only, never the tensors they describe.
use crate::native::Error;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

pub struct PinRegistry {
    pub limit: Cell<f64>,
    total: Cell<usize>,
    sizes: RefCell<BTreeMap<usize, usize>>,
}
pub trait PinBindings {
    type Tensor;
    fn allowed_type(&mut self, tensor: &Self::Tensor) -> Result<bool, Error>;
    fn is_cpu(&mut self, tensor: &Self::Tensor) -> Result<bool, Error>;
    fn is_pinned(&mut self, tensor: &Self::Tensor) -> Result<bool, Error>;
    fn is_contiguous(&mut self, tensor: &Self::Tensor) -> Result<bool, Error>;
    fn nbytes(&mut self, tensor: &Self::Tensor) -> Result<usize, Error>;
    fn extra_ram_release(&mut self) -> Result<(), Error>;
    fn ensure_registerable(&mut self, size: usize, evict_active: bool) -> Result<bool, Error>;
    fn pointer(&mut self, tensor: &Self::Tensor) -> Result<usize, Error>;
    fn register(&mut self, pointer: usize, size: usize) -> Result<bool, Error>;
    fn unregister(&mut self, pointer: usize) -> Result<bool, Error>;
    fn warning(&mut self, message: &'static str) -> Result<(), Error>;
    fn discard_async_error(&mut self) -> Result<(), Error>;
}
impl PinRegistry {
    pub fn new(limit: f64) -> Self {
        Self {
            limit: Cell::new(limit),
            total: Cell::new(0),
            sizes: RefCell::new(BTreeMap::new()),
        }
    }
    pub fn bytes(&self) -> usize {
        self.total.get()
    }
    pub fn registered_size(&self, pointer: usize) -> Option<usize> {
        self.sizes.borrow().get(&pointer).copied()
    }
    pub fn pin<B: PinBindings>(
        &self,
        tensor: &B::Tensor,
        evict_active: bool,
        bindings: &mut B,
    ) -> Result<bool, Error> {
        if self.limit.get() <= 0.
            || !bindings.allowed_type(tensor)?
            || !bindings.is_cpu(tensor)?
            || bindings.is_pinned(tensor)?
            || !bindings.is_contiguous(tensor)?
        {
            return Ok(false);
        }
        let size = bindings.nbytes(tensor)?;
        bindings.extra_ram_release()?;
        if !bindings.ensure_registerable(size, evict_active)? {
            return Ok(false);
        }
        let pointer = bindings.pointer(tensor)?;
        if pointer == 0 {
            return Ok(false);
        }
        if bindings.register(pointer, size)? {
            self.sizes.borrow_mut().insert(pointer, size);
            self.total.set(
                self.total
                    .get()
                    .checked_add(size)
                    .ok_or_else(|| overflow("pin total"))?,
            );
            return Ok(true);
        }
        bindings.warning("Pin error.")?;
        bindings.discard_async_error()?;
        Ok(false)
    }
    pub fn unpin<B: PinBindings>(
        &self,
        tensor: &B::Tensor,
        bindings: &mut B,
    ) -> Result<bool, Error> {
        if self.limit.get() <= 0. || !bindings.is_cpu(tensor)? {
            return Ok(false);
        }
        let pointer = bindings.pointer(tensor)?;
        let size = bindings.nbytes(tensor)?;
        let stored = self.registered_size(pointer);
        let Some(stored) = stored else {
            bindings.warning("Tried to unpin tensor not pinned by ComfyUI")?;
            return Ok(false);
        };
        if size != stored {
            bindings.warning("Size of pinned tensor changed")?;
            return Ok(false);
        }
        if bindings.unregister(pointer)? {
            // The external call may reenter. Resolve pop and total afterwards.
            let size = self.sizes.borrow_mut().remove(&pointer).ok_or_else(|| {
                Error::ReferenceException {
                    class: "KeyError",
                    message: pointer.to_string(),
                }
            })?;
            self.total.set(
                self.total
                    .get()
                    .checked_sub(size)
                    .ok_or_else(|| overflow("unpin total"))?,
            );
            return Ok(true);
        }
        bindings.warning("Unpin error.")?;
        bindings.discard_async_error()?;
        Ok(false)
    }
}
fn overflow(operation: &str) -> Error {
    Error::BindingGap {
        backend: "host_registration".into(),
        operation: format!("{operation} requires arbitrary precision source arithmetic"),
    }
}
