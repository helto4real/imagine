//! Native source-owner bindings. The state belongs to the canonical Patcher;
//! a backend view retains the same attribute store, not copied source fields.
use super::{Handle, Patcher};
use crate::{native::Error, patching::model_patcher::PatcherState};
use std::{any::Any, rc::Rc};

pub trait OwnerBindings<'a> {
    fn as_any(&self) -> &dyn Any;
    /// Actual source load_device field, including partial-constructor errors.
    fn load_device(&self, state: &PatcherState) -> Result<tch::Device, Error>;
    /// Original parent attribute access and canonical owner resolution.
    fn parent(&self, state: &PatcherState) -> Result<Option<Handle<'a>>, Error>;
    /// ModelPatcher.model_patches_models, not additional_models traversal.
    fn model_patches_models(&self, state: &PatcherState) -> Result<Vec<Handle<'a>>, Error>;
    /// Actual source-class destructor body, before weak callbacks. Model
    /// unpin/ejection/detachment and dynamic registration checks stay required.
    fn finalize(&self, owner: &Patcher<'a>) -> Result<(), Error>;
    /// Native counterpart of an exception in a finalizer. No panic/default
    /// silent suppression or caller-selected error policy is supplied here.
    fn unraisable(&self, error: Error);
}
pub(crate) struct SourceOwner<'a> {
    pub state: PatcherState,
    pub bindings: Rc<dyn OwnerBindings<'a> + 'a>,
}
