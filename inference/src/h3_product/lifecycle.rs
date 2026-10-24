use crate::{
    components::{ComponentBuffers, StaticModel},
    memory::{Handle, Manager},
    native::Error,
};
use std::rc::Rc;

/// Finish a component whose model will not be reused. Unloading a live model
/// would first copy its resident weights to CPU, only to discard them here.
pub fn release_component<'a>(
    model: Rc<StaticModel<'a, ComponentBuffers>>,
    patcher: Handle<'a>,
    manager: &mut Manager<'a>,
) -> Result<(), Error> {
    // Report unpin failures explicitly; destructor-only cleanup cannot return them.
    let unpin = model.owner.borrow_mut().unpin_all_weights();
    drop(patcher);
    drop(model);
    let registrations = manager.cleanup();
    unpin.and(registrations)
}
