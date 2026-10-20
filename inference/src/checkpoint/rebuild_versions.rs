//! Dense native kernel of _rebuild_tensor_v2/v3, before fake-device restoration.
//! Raw Python argument conversion and TypedStorage/global object construction
//! belong to the checkpoint object binding, not this typed kernel.
use super::storage_rebuild;
use crate::{native::Error, patching::options::Options, source_exception::Frame};
use std::rc::Rc;
use tch::Tensor;
#[derive(Clone, Copy)]
pub enum Version {
    V2,
    V3,
}
pub struct Arguments<'a> {
    pub storage: &'a Tensor,
    pub dtype: i32,
    pub offset: i64,
    pub size: &'a [i64],
    pub stride: &'a [i64],
    pub requires_grad: bool,
    pub backward_hooks: Options,
    /// Already converted string/bool map. Presence sets flags, even for false.
    pub metadata: &'a [(String, bool)],
}
pub struct Rebuilt {
    pub tensor: Tensor,
    pub backward_hooks: Options,
}
/// The caller must retain this complete object, including its hook owner, and
/// apply the source fake-device restoration using its native mode context.
pub fn dense(version: Version, args: Arguments<'_>) -> Result<Rebuilt, Error> {
    let mut partial: Option<Tensor> = None;
    let result = (|| {
        match version {
            Version::V2 => {
                partial = Some(storage_rebuild::tensor(
                    args.storage,
                    args.dtype,
                    args.offset,
                    args.size,
                    args.stride,
                )?);
                storage_rebuild::requires_grad(partial.as_ref().unwrap(), args.requires_grad)?;
            }
            Version::V3 => {
                partial = Some(storage_rebuild::empty(
                    args.storage,
                    args.dtype,
                    args.requires_grad,
                )?);
                storage_rebuild::assign_storage(
                    partial.as_ref().unwrap(),
                    args.storage,
                    args.offset,
                    args.size,
                    args.stride,
                )?;
            }
        }
        let has = |key: &str| args.metadata.iter().any(|(name, _)| name == key);
        storage_rebuild::metadata_flags(partial.as_ref().unwrap(), has("conj"), has("neg"))?;
        Ok(())
    })();
    result.map_err(|error: Error| {
        let frame = Frame::new(match version {
            Version::V2 => "_rebuild_tensor_v2",
            Version::V3 => "_rebuild_tensor_v3",
        });
        frame.set("storage", Rc::new(args.storage.shallow_clone()));
        frame.set("storage_offset", Rc::new(args.offset));
        frame.set("size", Rc::new(args.size.to_vec()));
        frame.set("stride", Rc::new(args.stride.to_vec()));
        frame.set("requires_grad", Rc::new(args.requires_grad));
        frame.set("backward_hooks", Rc::new(args.backward_hooks.clone()));
        frame.set("metadata", Rc::new(args.metadata.to_vec()));
        if let Version::V3 = version {
            frame.set("dtype", Rc::new(args.dtype));
        }
        if let Some(tensor) = &partial {
            frame.set(
                match version {
                    Version::V2 => "tensor",
                    Version::V3 => "t",
                },
                Rc::new(tensor.shallow_clone()),
            );
        }
        frame.finish();
        error.retain_frame(frame)
    })?;
    Ok(Rebuilt {
        tensor: partial.unwrap(),
        backward_hooks: args.backward_hooks,
    })
}
