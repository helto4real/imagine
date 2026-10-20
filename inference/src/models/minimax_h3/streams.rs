//! Typed shared AV stream owners and original CPU prepare_noise behavior.
//! The source class is a mutable Python list wrapper, not a Torch nested tensor.
use crate::{
    execution_value::Value,
    native::{self, Error},
};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Kind, Tensor};
struct Object {
    tensors: Rc<RefCell<Vec<Value>>>,
}
#[derive(Clone)]
pub struct NestedTensor(Rc<Object>);
impl NestedTensor {
    pub fn new(tensors: Vec<Value>) -> Self {
        Self(Rc::new(Object {
            tensors: Rc::new(RefCell::new(tensors)),
        }))
    }
    pub fn same_object(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub fn unbind(&self) -> Rc<RefCell<Vec<Value>>> {
        self.0.tensors.clone()
    }
    pub fn shallow_copy(&self) -> Self {
        Self::new(self.0.tensors.borrow().clone())
    }
    pub fn scale_audio(&self, scale: f64) -> Result<Self, Error> {
        if scale == 1.0 {
            return Ok(self.clone());
        }
        let streams = self.0.tensors.borrow();
        let first = streams.first().ok_or_else(index_error)?.clone();
        let audio = Value::tensor(
            streams
                .get(1)
                .ok_or_else(index_error)?
                .as_tensor()?
                .f_mul_scalar(scale)?,
        );
        let mut output = vec![first, audio];
        output.extend_from_slice(&streams[2..]);
        Ok(Self::new(output))
    }
    /// Seeds the shared CPU generator once before iterating all streams. The
    /// native manual_seed call also preserves the standard Torch device side effects.
    pub fn prepare_noise(&self, seed: u64, indices: Option<&[i64]>) -> Result<Self, Error> {
        tch::manual_seed(seed as i64);
        let mut output = vec![];
        for value in self.0.tensors.borrow().iter() {
            let source = value.as_tensor()?;
            let noise = match indices {
                None => Tensor::f_randn(source.size(), (Kind::Float, Device::Cpu))?
                    .f_to_kind(source.kind())?,
                Some(indices) => {
                    let mut unique = indices.to_vec();
                    unique.sort_unstable();
                    unique.dedup();
                    let last = unique.last().ok_or_else(|| Error::ReferenceException {
                        class: "IndexError",
                        message: "index -1 is out of bounds for axis 0 with size 0".into(),
                    })?;
                    let mut generated = vec![];
                    let mut shape = vec![1];
                    shape.extend(source.size().into_iter().skip(1));
                    for i in 0..=*last {
                        let noise = Tensor::f_randn(&shape, (Kind::Float, Device::Cpu))?
                            .f_to_kind(source.kind())?;
                        if unique.binary_search(&i).is_ok() {
                            generated.push(noise);
                        }
                    }
                    let mut ordered = vec![];
                    for i in indices {
                        let index = unique
                            .binary_search(i)
                            .expect("deduplicated original index");
                        ordered.push(
                            generated
                                .get(index)
                                .ok_or_else(index_error)?
                                .shallow_clone(),
                        );
                    }
                    Tensor::f_cat(&ordered, 0)?
                }
            };
            output.push(Value::tensor(noise));
        }
        Ok(Self::new(output))
    }
    pub fn prepare_empty_noise(&self, native: &native::Kitchen) -> Result<Self, Error> {
        let mut output = vec![];
        for value in self.0.tensors.borrow().iter() {
            let result = unsafe {
                crate::boxed_aten::call(
                    native,
                    "aten::zeros_like",
                    "",
                    std::slice::from_ref(value),
                    &vec![("device".into(), Value::Device(Device::Cpu))],
                )?
            };
            output.push(result);
        }
        Ok(Self::new(output))
    }
    pub fn pack(&self) -> Result<(Value, crate::context_windows::LatentShapes), Error> {
        crate::context_windows::pack_latents(&self.0.tensors.borrow())
    }
    pub fn unpack(
        combined: &Value,
        shapes: &crate::context_windows::LatentShapes,
    ) -> Result<Self, Error> {
        Ok(Self::new(crate::context_windows::unpack_latents(
            combined, shapes,
        )?))
    }
    pub fn rng_state() -> Result<Tensor, Error> {
        native::global_rng_state(Device::Cpu)
    }
}
fn index_error() -> Error {
    Error::ReferenceException {
        class: "IndexError",
        message: "list index out of range".into(),
    }
}
