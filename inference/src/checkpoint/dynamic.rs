//! Comfy dynamic cast modules: CPU backing and operation-scoped VBAR residency.
use super::{
    PreparedParameters,
    dynamic_vbar::{Fault, Range, Vbar},
};
use crate::{execution::Runtime, execution_value::Value, native::Error, storage::QuantizedStorage};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use tch::{Device, Kind, Tensor};

fn text_layer(name: &str) -> Option<usize> {
    name.strip_prefix("model.layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}
fn text_module_order(
    modules: Vec<crate::models::qwen_image21::OffloadModule>,
) -> Vec<crate::models::qwen_image21::OffloadModule> {
    let mut indexed = modules.into_iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(
        |(ai, a), (bi, b)| match (text_layer(&a.name), text_layer(&b.name)) {
            // SD1ClipModel forwards Llama's layer units. root.modules() retains
            // constructor registration order inside each unit, not size order.
            (Some(a), Some(b)) => (a, ai).cmp(&(b, bi)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => (
                a.offload_bytes >= 64 * 1024,
                std::cmp::Reverse(a.offload_bytes),
                a.module_bytes,
                &a.name,
            )
                .cmp(&(
                    b.offload_bytes >= 64 * 1024,
                    std::cmp::Reverse(b.offload_bytes),
                    b.module_bytes,
                    &b.name,
                )),
        },
    );
    indexed.into_iter().map(|(_, m)| m).collect()
}

fn aligned(bytes: usize) -> usize {
    (bytes + 1023) & !1023
}
fn tensors(value: &Value) -> Result<Vec<Tensor>, Error> {
    if let Ok(q) = value.as_quantized() {
        let mut out = vec![q.payload];
        out.extend(
            q.params
                .field_order
                .iter()
                .filter_map(|k| q.params.tensors.get(k))
                .map(Tensor::shallow_clone),
        );
        Ok(out)
    } else {
        Ok(vec![value.as_tensor()?])
    }
}
fn packed_size(values: &[Value]) -> Result<u64, Error> {
    Ok(values
        .iter()
        .map(tensors)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .flatten()
        .map(|t| aligned(t.numel() * t.kind().elt_size_in_bytes()) as u64)
        .sum())
}
fn gather(rt: &Runtime, originals: &[Value], buffer: &Tensor) -> Result<Vec<Value>, Error> {
    let mut offset = 0;
    let mut copy = |source: &Tensor| -> Result<Tensor, Error> {
        let bytes = source.numel() * source.kind().elt_size_in_bytes();
        let mut view = buffer
            .f_narrow(0, offset, bytes as i64)?
            .f_view_dtype(source.kind())?
            .f_view(source.size().as_slice())?;
        rt.native.copy_tensor(&mut view, source, true)?;
        offset += aligned(bytes) as i64;
        Ok(view)
    };
    originals
        .iter()
        .map(|value| {
            if let Ok(q) = value.as_quantized() {
                let payload = copy(&q.payload)?;
                let mut params = q.params.shallow_copy();
                for key in &q.params.field_order {
                    if let Some(tensor) = q.params.tensors.get(key) {
                        params.tensors.insert(key.clone(), copy(tensor)?);
                    }
                }
                Ok(Value::quantized(QuantizedStorage::from_validated_parts(
                    payload, q.layout, params,
                )))
            } else {
                Ok(Value::tensor(copy(&value.as_tensor()?)?))
            }
        })
        .collect()
}
struct Cached {
    signature: Vec<u32>,
    values: Vec<Value>,
}
struct Owner {
    vbar: Arc<Vbar>,
    native: crate::native::Kitchen,
}
impl Drop for Owner {
    fn drop(&mut self) {
        // Also covers constructor/forward errors before the explicit successful
        // component boundary. The library outlives all non-owning CUDA views.
        let _ = self.native.synchronize_device(0);
    }
}
pub struct Module {
    original: Vec<Value>,
    cached: Mutex<Option<Cached>>,
    range: Range,
    owner: Arc<Owner>,
}
pub struct Lease {
    pub values: Vec<Value>,
    // Values must drop before the pin and its owner, including error paths.
    _buffer: Option<Tensor>,
    _fault: Fault,
    _owner: Arc<Owner>,
}
impl Module {
    pub fn resident_model_bytes(&self) -> usize {
        self.owner.vbar.loaded_size()
    }
    pub fn acquire(&self, rt: &Runtime) -> Result<Lease, Error> {
        let fault = self
            .owner
            .vbar
            .acquire(self.range.offset, self.range.size)?;
        let mut cache = self.cached.lock().unwrap();
        if let Some(signature) = &fault.signature
            && let Some(cached) = cache.as_ref()
            && cached.signature == *signature
        {
            return Ok(Lease {
                values: cached.values.clone(),
                _buffer: None,
                _fault: fault,
                _owner: self.owner.clone(),
            });
        }
        // AIMDO returns OOM for a range that cannot be pinned. Comfy uses an
        // ordinary temporary tensor for this operation in that case.
        let buffer = if fault.signature.is_some() {
            unsafe {
                Tensor::f_from_blob(
                    self.owner.vbar.address(self.range.offset),
                    &[self.range.size as i64],
                    &[],
                    Kind::Uint8,
                    Device::Cuda(0),
                )?
            }
        } else {
            Tensor::f_empty([self.range.size as i64], (Kind::Uint8, Device::Cuda(0)))?
        };
        let values = gather(rt, &self.original, &buffer)?;
        if let Some(signature) = &fault.signature {
            *cache = Some(Cached {
                signature: signature.clone(),
                values: values.clone(),
            });
        } else {
            *cache = None;
        }
        Ok(Lease {
            values,
            _buffer: Some(buffer),
            _fault: fault,
            _owner: self.owner.clone(),
        })
    }
}

/// The returned parameters own the model VBAR through every dynamic module.
/// Synchronize its device before dropping the final encoder/parameter owner.
pub fn prepare_text(
    rt: &Runtime,
    library: &Path,
    mut parameters: PreparedParameters,
    inventory: crate::models::qwen_image21::ComponentInventory,
) -> Result<PreparedParameters, Error> {
    let vbar = unsafe { Vbar::load(library, 0, inventory.model_size * 10)? };
    vbar.prioritize();
    let owner = Arc::new(Owner {
        vbar,
        native: rt.native.clone(),
    });
    for module in text_module_order(inventory.modules) {
        if module.cast_weights && module.module_bytes > 16 * 1024 {
            let original = ["weight", "bias"]
                .iter()
                .filter_map(|k| {
                    parameters
                        .values
                        .get(&format!("{}.{}", module.name, k))
                        .cloned()
                })
                .collect::<Vec<_>>();
            let range = owner.vbar.allocate(packed_size(&original)?)?;
            parameters.dynamic_modules.insert(
                module.name.clone(),
                Arc::new(Module {
                    original,
                    cached: Mutex::new(None),
                    range,
                    owner: owner.clone(),
                }),
            );
            for key in module
                .parameters
                .iter()
                .filter(|k| !k.ends_with(".weight") && !k.ends_with(".bias"))
            {
                preload(rt, &mut parameters, key)?;
            }
        } else {
            for key in &module.parameters {
                preload(rt, &mut parameters, key)?;
            }
        }
    }
    for buffer in inventory.buffers {
        if let Some(value) = buffer.initial_value {
            let kind = if buffer.default_dtype {
                rt.native.default_dtype()?
            } else {
                Kind::Float
            };
            parameters.values.insert(
                buffer.name.clone(),
                Value::tensor(
                    Tensor::f_full([], value, (kind, Device::Cpu))?
                        .f_set_requires_grad(buffer.requires_grad)?,
                ),
            );
        }
        preload(rt, &mut parameters, &buffer.name)?;
    }
    Ok(parameters)
}
fn preload(rt: &Runtime, parameters: &mut PreparedParameters, key: &str) -> Result<(), Error> {
    if let Some(value) = parameters.values.get(key) {
        let value = if let Ok(q) = value.as_quantized() {
            let mut target = q.empty_like(&rt.native, Some(Device::Cuda(0)), None)?;
            target.copy_from(&rt.native, &q, true)?;
            Value::quantized(target)
        } else {
            Value::tensor(value.as_tensor()?.f_to_device(Device::Cuda(0))?)
        };
        parameters.values.insert(key.into(), value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packing_matches_comfy_alignment() {
        assert_eq!(
            [0, 1, 1023, 1024, 1025].map(aligned),
            [0, 1024, 1024, 1024, 2048]
        );
        let values = vec![
            Value::tensor(Tensor::zeros([7], (Kind::Half, Device::Cpu))),
            Value::tensor(Tensor::zeros([600], (Kind::Float, Device::Cpu))),
        ];
        assert_eq!(packed_size(&values).unwrap(), 4096);
    }
    #[test]
    fn language_units_precede_dynamic_size_sorted_remainder() {
        let module = |name: &str, size| crate::models::qwen_image21::OffloadModule {
            name: name.into(),
            module_bytes: size,
            offload_bytes: size,
            cast_weights: true,
            parameters: vec![],
        };
        let modules = vec![
            module("embed", 100_000),
            module("model.layers.0.q", 100),
            module("model.layers.0.k", 200),
            module("model.layers.1.q", 300),
            module("tiny", 10),
            module("large", 200_000),
        ];
        assert_eq!(
            text_module_order(modules)
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
            [
                "model.layers.0.q",
                "model.layers.0.k",
                "model.layers.1.q",
                "tiny",
                "large",
                "embed"
            ]
        );
    }
}
