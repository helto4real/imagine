//! Layout-independent QuantizedTensor storage operations from Kitchen base.py.
//!
//! These operate on validated layout parts; layout-specific quantization,
//! packing, dequantization and matmul are separate operations.
use crate::execution_value::Value;
use crate::native::{Error, Kitchen};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarField {
    None,
    Bool(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    Shape(Vec<i64>),
}
pub struct LayoutParams {
    pub original_kind: Kind,
    pub original_shape: Vec<i64>,
    /// Actual tensor fields returned by the layout's `_tensor_fields()`.
    pub tensors: BTreeMap<String, Tensor>,
    pub fields: BTreeMap<String, ScalarField>,
    /// Dataclass declaration order, including orig_dtype/orig_shape and tensors.
    pub field_order: Vec<String>,
}
impl LayoutParams {
    pub(crate) fn shallow_copy(&self) -> Self {
        Self {
            original_kind: self.original_kind,
            original_shape: self.original_shape.clone(),
            tensors: self
                .tensors
                .iter()
                .map(|(k, v)| (k.clone(), v.shallow_clone()))
                .collect(),
            fields: self.fields.clone(),
            field_order: self.field_order.clone(),
        }
    }
    fn deep_clone(&self, native: &Kitchen) -> Result<Self, Error> {
        Ok(Self {
            original_kind: self.original_kind,
            original_shape: self.original_shape.clone(),
            tensors: self
                .field_order
                .iter()
                .filter_map(|name| self.tensors.get(name).map(|tensor| (name, tensor)))
                .map(|(k, v)| Ok((k.clone(), native.clone_tensor(v)?)))
                .collect::<Result<_, Error>>()?,
            fields: self.fields.clone(),
            field_order: self.field_order.clone(),
        })
    }
    fn to_device(&self, device: Device) -> Result<Self, Error> {
        Ok(Self {
            original_kind: self.original_kind,
            original_shape: self.original_shape.clone(),
            tensors: self
                .field_order
                .iter()
                .filter_map(|name| self.tensors.get(name).map(|tensor| (name, tensor)))
                .map(|(k, v)| Ok((k.clone(), v.f_to_device(device)?)))
                .collect::<Result<_, Error>>()?,
            fields: self.fields.clone(),
            field_order: self.field_order.clone(),
        })
    }
}
pub struct QuantizedStorage {
    pub payload: Tensor,
    pub layout: String,
    pub params: LayoutParams,
    tensor_objects: Arc<Mutex<BTreeMap<String, (u64, Value)>>>,
}
pub enum StorageResult<'a> {
    Borrowed(&'a QuantizedStorage),
    Owned(QuantizedStorage),
}
impl AsRef<QuantizedStorage> for StorageResult<'_> {
    fn as_ref(&self) -> &QuantizedStorage {
        match self {
            Self::Borrowed(x) => x,
            Self::Owned(x) => x,
        }
    }
}
impl QuantizedStorage {
    pub fn from_validated_parts(payload: Tensor, layout: String, params: LayoutParams) -> Self {
        Self {
            payload,
            layout,
            params,
            tensor_objects: Arc::default(),
        }
    }
    pub(crate) fn with_parts(
        &self,
        native: &Kitchen,
        payload: Tensor,
        params: LayoutParams,
    ) -> Self {
        let objects = self
            .tensor_objects
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(name, (id, value))| {
                params
                    .tensors
                    .get(name)
                    .filter(|t| native.tensor_identity(t) == *id)
                    .map(|_| (name.clone(), (*id, value.clone())))
            })
            .collect();
        Self {
            payload,
            layout: self.layout.clone(),
            params,
            tensor_objects: Arc::new(Mutex::new(objects)),
        }
    }
    /// Keep the original ordinary tensor object (and its attributes) with this field.
    pub fn tensor_field_value(&self, native: &Kitchen, name: &str) -> Option<Value> {
        let tensor = self.params.tensors.get(name)?;
        let id = native.tensor_identity(tensor);
        let mut objects = self.tensor_objects.lock().unwrap();
        if let Some((old_id, value)) = objects.get(name)
            && *old_id == id
        {
            return Some(value.clone());
        }
        let value = objects
            .values()
            .find(|(old_id, _)| *old_id == id)
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| Value::tensor(tensor.shallow_clone()));
        objects.insert(name.into(), (id, value.clone()));
        Some(value)
    }
    pub fn storage_shape(&self) -> Vec<i64> {
        self.payload.size()
    }
    pub fn storage_kind(&self) -> Kind {
        self.payload.kind()
    }
    pub fn storage_bytes(&self) -> usize {
        self.payload.numel() * self.payload.kind().elt_size_in_bytes()
    }
    pub fn detach(&self, native: &Kitchen) -> Result<Self, Error> {
        Ok(self.with_parts(
            native,
            self.payload.f_detach()?,
            self.params.deep_clone(native)?,
        ))
    }
    pub fn clone_storage(&self, native: &Kitchen) -> Result<Self, Error> {
        Ok(self.with_parts(
            native,
            native.clone_tensor(&self.payload)?,
            self.params.deep_clone(native)?,
        ))
    }
    pub fn contiguous(&self, native: &Kitchen) -> Result<StorageResult<'_>, Error> {
        if self.payload.is_contiguous() {
            Ok(StorageResult::Borrowed(self))
        } else {
            Ok(StorageResult::Owned(self.with_parts(
                native,
                self.payload.f_contiguous()?,
                self.params.deep_clone(native)?,
            )))
        }
    }
    pub fn to(
        &self,
        native: &Kitchen,
        device: Option<Device>,
        kind: Option<Kind>,
        force_copy: bool,
    ) -> Result<StorageResult<'_>, Error> {
        let needs_device = device.is_some_and(|x| x != self.payload.device());
        let needs_kind = kind.is_some_and(|x| x != self.params.original_kind);
        if !needs_device && !needs_kind && !force_copy {
            return Ok(StorageResult::Borrowed(self));
        }
        let (payload, mut params) = if needs_device {
            let device = device.unwrap();
            (
                self.payload.f_to_device(device)?,
                self.params.to_device(device)?,
            )
        } else {
            (
                if force_copy {
                    native.clone_tensor(&self.payload)?
                } else {
                    self.payload.shallow_clone()
                },
                self.params.deep_clone(native)?,
            )
        };
        if let Some(kind) = kind {
            params.original_kind = kind;
        }
        Ok(StorageResult::Owned(
            self.with_parts(native, payload, params),
        ))
    }
    pub fn to_device_spec(
        &self,
        native: &Kitchen,
        device: Option<crate::native::DeviceSpec>,
        kind: Option<Kind>,
        force_copy: bool,
    ) -> Result<StorageResult<'_>, Error> {
        if let Some(device) = device.filter(|d| !d.equals_tensor_device(self.payload.device())) {
            let payload = native.to_device_spec(&self.payload, device)?;
            let mut params = self.params.shallow_copy();
            for name in &self.params.field_order {
                if let Some(t) = self.params.tensors.get(name) {
                    params
                        .tensors
                        .insert(name.clone(), native.to_device_spec(t, device)?);
                }
            }
            if let Some(kind) = kind {
                params.original_kind = kind;
            }
            Ok(StorageResult::Owned(
                self.with_parts(native, payload, params),
            ))
        } else {
            self.to(native, None, kind, force_copy)
        }
    }
    pub fn empty_like_device_spec(
        &self,
        native: &Kitchen,
        device: Option<crate::native::DeviceSpec>,
        kind: Option<Kind>,
    ) -> Result<Self, Error> {
        let Some(device) = device else {
            return self.empty_like(native, None, kind);
        };
        let payload = native.empty_like_device_spec(&self.payload, device)?;
        let mut params = self.params.deep_clone(native)?;
        for name in &self.params.field_order {
            if let Some(t) = params.tensors.get(name) {
                let moved = native.to_device_spec(t, device)?;
                params.tensors.insert(name.clone(), moved);
            }
        }
        if let Some(kind) = kind {
            params.original_kind = kind;
        }
        Ok(self.with_parts(native, payload, params))
    }
    pub fn empty_like(
        &self,
        native: &Kitchen,
        device: Option<Device>,
        kind: Option<Kind>,
    ) -> Result<Self, Error> {
        let payload = native.empty_like(&self.payload, device.unwrap_or(self.payload.device()))?;
        let mut params = self.params.deep_clone(native)?;
        if let Some(device) = device {
            params = params.to_device(device)?;
        }
        if let Some(kind) = kind {
            params.original_kind = kind;
        }
        Ok(self.with_parts(native, payload, params))
    }
    pub fn copy_from(
        &mut self,
        native: &Kitchen,
        source: &Self,
        non_blocking: bool,
    ) -> Result<(), Error> {
        if self.layout != source.layout {
            return Err(Error::Native("Quantized layout mismatch".into()));
        }
        native.copy_tensor(&mut self.payload, &source.payload, non_blocking)?;
        let original_kind = self.params.original_kind;
        for name in &self.params.field_order {
            match name.as_str() {
                "orig_dtype" => self.params.original_kind = source.params.original_kind,
                "orig_shape" => self.params.original_shape = source.params.original_shape.clone(),
                _ if self.params.tensors.contains_key(name) => {
                    let source = source.params.tensors.get(name).ok_or_else(|| {
                        Error::Native(format!("Missing layout tensor field {name}"))
                    })?;
                    native.copy_tensor(
                        self.params.tensors.get_mut(name).unwrap(),
                        source,
                        non_blocking,
                    )?;
                }
                _ => {
                    let value = source
                        .params
                        .fields
                        .get(name)
                        .ok_or_else(|| Error::Native(format!("Missing layout field {name}")))?;
                    self.params.fields.insert(name.clone(), value.clone());
                }
            }
        }
        // copy_ retains the destination's logical dtype, as in the reference.
        self.params.original_kind = original_kind;
        Ok(())
    }
    pub fn shallow_parts(&self) -> Self {
        Self {
            payload: self.payload.shallow_clone(),
            layout: self.layout.clone(),
            params: self.params.shallow_copy(),
            tensor_objects: self.tensor_objects.clone(),
        }
    }
}
