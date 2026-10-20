//! Common native component loading before model-family construction.
//! No checkpoint-wide dtype cast, invented model keys or legacy-metadata skip.
use super::{Aimdo, Safetensors, state_dict};
use crate::{
    execution::Runtime,
    execution_value::Value,
    native::{self, Error},
    patching::options::Options,
};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    sync::Arc,
};
struct Annotation {
    identity: usize,
    storage: native::WeakStorage,
    path: Arc<PathBuf>,
}
/// Source UntypedStorage annotations follow actual StorageImpl aliases and do
/// not keep tensor data alive. One execution owner shares this service across
/// components, so annotating a shared storage replaces its previous path.
#[derive(Default)]
pub struct StorageAnnotations(RefCell<Vec<Annotation>>);
impl StorageAnnotations {
    pub fn annotate(
        &self,
        tensor: &tch::Tensor,
        canonical_path: Arc<PathBuf>,
    ) -> Result<(), Error> {
        let identity = native::storage_info(tensor)?.0;
        let mut entries = self.0.borrow_mut();
        entries.retain(|entry| !entry.storage.expired());
        if let Some(entry) = entries.iter_mut().find(|entry| entry.identity == identity) {
            entry.path = canonical_path;
        } else {
            entries.push(Annotation {
                identity,
                storage: native::WeakStorage::new(tensor)?,
                path: canonical_path,
            });
        }
        Ok(())
    }
    pub fn source_path(&self, tensor: &tch::Tensor) -> Result<Option<Arc<PathBuf>>, Error> {
        let identity = native::storage_info(tensor)?.0;
        let mut entries = self.0.borrow_mut();
        entries.retain(|entry| !entry.storage.expired());
        Ok(entries
            .iter()
            .find(|entry| entry.identity == identity)
            .map(|entry| entry.path.clone()))
    }
}
pub struct ComponentState {
    pub weights: Options,
    pub metadata: Options,
}
/// The source callers normalize only when custom_operations is exactly None,
/// including when another supplied value is false. Keep both returned owners.
///
/// # Safety
/// The runtime and tensor values must satisfy the frozen ATen/Kitchen ABI.
pub unsafe fn normalize(
    rt: &Runtime,
    weights: Options,
    metadata: Options,
    custom_operations: &Options,
) -> Result<ComponentState, Error> {
    let (weights, metadata) = if matches!(custom_operations, Options::Atom(Value::None)) {
        unsafe { state_dict::convert_old_quants(rt, &weights, "", &metadata)? }
    } else {
        (weights, metadata)
    };
    Ok(ComponentState { weights, metadata })
}
/// Explicit AIMDO branch. The execution owner chooses the same loader policy
/// as the reference; this function never guesses or falls back to another one.
///
/// # Safety
/// The runtime and AIMDO library must satisfy the frozen native ABIs.
pub unsafe fn load_aimdo(
    rt: &Runtime,
    aimdo: &Aimdo,
    path: &Path,
    custom_operations: &Options,
    annotations: &StorageAnnotations,
) -> Result<ComponentState, Error> {
    let loaded = Safetensors::load_aimdo(path, aimdo, &rt.native)?;
    let metadata = loaded.source_metadata.source_options()?;
    let mut weights = vec![];
    let mut canonical = None::<Arc<PathBuf>>;
    for (name, entry) in loaded.tensors {
        let path = canonical
            .get_or_insert_with(|| Arc::new(entry.source_path.clone()))
            .clone();
        annotations.annotate(&entry.tensor, path)?;
        weights.push((name, Options::Atom(Value::tensor(entry.tensor))));
    }
    unsafe {
        normalize(
            rt,
            Options::dictionary(weights),
            metadata,
            custom_operations,
        )
    }
}
