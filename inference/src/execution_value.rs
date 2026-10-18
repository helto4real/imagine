//! Owned tensor identities and ordered recursive arguments for Kitchen dispatch.
use crate::{native::Error, registry::Argument, storage::QuantizedStorage};
use std::sync::{Arc, Mutex};
use tch::{Device, Kind, Tensor};

/// Tensor-owned attributes have the same lifetime as this ordinary tensor object.
pub struct TensorObject {
    pub tensor: Tensor,
    pub(crate) lora_up: Option<crate::svdquant::LoraUp>,
}
impl std::ops::Deref for TensorObject {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.tensor
    }
}

pub type Kwargs = Vec<(String, Value)>;
#[derive(Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
    DType(Kind),
    Device(Device),
    Tensor(Arc<Mutex<TensorObject>>),
    Quantized(Arc<Mutex<QuantizedStorage>>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Kwargs),
}
impl Value {
    pub fn tensor(t: Tensor) -> Self {
        Self::Tensor(Arc::new(Mutex::new(TensorObject {
            tensor: t,
            lora_up: None,
        })))
    }
    pub fn quantized(q: QuantizedStorage) -> Self {
        Self::Quantized(Arc::new(Mutex::new(q)))
    }
    /// Original per-object `_ck_natural_lora_up` attribute, when populated.
    pub fn natural_lora_cache(&self) -> Option<Tensor> {
        match self {
            Self::Tensor(t) => t
                .lock()
                .unwrap()
                .lora_up
                .as_ref()
                .and_then(|up| up.cached_natural())
                .map(Tensor::shallow_clone),
            _ => None,
        }
    }
    pub fn as_tensor(&self) -> Result<Tensor, Error> {
        match self {
            Self::Tensor(t) => Ok(t.lock().unwrap().shallow_clone()),
            _ => Err(type_error("Expected an ordinary Tensor")),
        }
    }
    /// Original raw CUDA export boundary, with D-04-NULL-DLPACK-EXPORT.
    pub fn as_cuda_export(&self) -> Result<Tensor, Error> {
        if let Self::Quantized(q) = self {
            let q = q.lock().unwrap();
            if !q.params.original_shape.contains(&0) {
                return Err(type_error(
                    "Cannot export a non-empty QuantizedTensor with null DLPack data to a CUDA kernel.",
                ));
            }
            return Ok(Tensor::f_empty(
                q.params.original_shape.as_slice(),
                (q.params.original_kind, q.payload.device()),
            )?);
        }
        self.as_tensor()
    }
    pub fn as_quantized(&self) -> Result<QuantizedStorage, Error> {
        match self {
            Self::Quantized(q) => Ok(q.lock().unwrap().shallow_parts()),
            _ => Err(type_error("Expected QuantizedTensor")),
        }
    }
    pub fn same_object(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Tensor(a), Self::Tensor(b)) => Arc::ptr_eq(a, b),
            (Self::Quantized(a), Self::Quantized(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    pub fn metadata(&self, native: &crate::native::Kitchen) -> Result<Argument<'static>, Error> {
        self.metadata_with(native, None)
    }
    pub fn metadata_for_call<'rt>(
        &self,
        runtime: &'rt crate::execution::Runtime,
    ) -> Result<Argument<'rt>, Error> {
        self.metadata_with(&runtime.native, Some(runtime))
    }
    fn metadata_with<'rt>(
        &self,
        native: &crate::native::Kitchen,
        runtime: Option<&'rt crate::execution::Runtime>,
    ) -> Result<Argument<'rt>, Error> {
        let metadata = self.metadata_snapshot(native)?;
        if matches!(
            self,
            Self::Tensor(_)
                | Self::Quantized(_)
                | Self::Device(_)
                | Self::Tuple(_)
                | Self::List(_)
                | Self::Dict(_)
        ) {
            return Ok(Argument::Source(crate::registry_source::SourceArgument {
                value: self.clone(),
                native: native.clone(),
                metadata: metadata.map(Box::new),
                runtime,
            }));
        }
        Ok(metadata.unwrap())
    }
    fn metadata_snapshot(
        &self,
        native: &crate::native::Kitchen,
    ) -> Result<Option<Argument<'static>>, Error> {
        Ok(Some(match self {
            Self::None => Argument::None,
            Self::Bool(v) => Argument::BoolValue(*v),
            Self::Int(v) => Argument::Integer(*v),
            Self::UInt(v) => Argument::Unsigned(*v),
            Self::Float(v) => Argument::FloatValue(*v),
            Self::Text(v) => Argument::Text(v.clone()),
            Self::DType(k) => Argument::DType(*k),
            Self::Tensor(t) => {
                let t = t.lock().unwrap();
                if t.f_kind().is_err() {
                    let kind = match native.tensor_type(&t)?.scalar_type {
                        44 => crate::native::ShellDType::E8M0,
                        45 => crate::native::ShellDType::PackedE2M1,
                        _ => {
                            return Err(Error::BindingGap {
                                backend: "metadata".into(),
                                operation: "unknown dtype".into(),
                            });
                        }
                    };
                    return Ok(Some(Argument::ShellTensor {
                        kind,
                        device: device_name(t.device()),
                        shape: t.size(),
                    }));
                }
                Argument::Tensor {
                    kind: t.kind(),
                    device: device_name(t.device()),
                    shape: t.size(),
                }
            }
            Self::Quantized(q) => {
                let q = q.lock().unwrap();
                Argument::Tensor {
                    kind: q.params.original_kind,
                    device: device_name(q.payload.device()),
                    shape: q.params.original_shape.clone(),
                }
            }
            Self::Tuple(_) | Self::List(_) | Self::Dict(_) | Self::Device(_) => return Ok(None),
        }))
    }
}
pub fn device_name(d: Device) -> String {
    match d {
        Device::Cpu => "cpu".into(),
        Device::Cuda(_) => "cuda".into(),
        Device::Mps => "mps".into(),
        Device::Vulkan => "vulkan".into(),
    }
}
pub fn type_error(s: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "TypeError",
        message: s.into(),
    }
}
pub fn get<'a>(
    args: &'a [Value],
    kwargs: &'a Kwargs,
    index: usize,
    name: &str,
) -> Option<&'a Value> {
    args.get(index)
        .or_else(|| kwargs.iter().find(|(k, _)| k == name).map(|(_, v)| v))
}

pub(crate) fn bind_arguments(
    name: &str,
    names: &[&str],
    required: usize,
    args: &[Value],
    kwargs: &Kwargs,
) -> Result<(), Error> {
    let mut provided = (0..names.len()).map(|i| i < args.len()).collect::<Vec<_>>();
    for (name_kw, _) in kwargs {
        let Some(i) = names.iter().position(|n| *n == name_kw) else {
            return Err(type_error(format!(
                "{name}() got an unexpected keyword argument '{name_kw}'"
            )));
        };
        if provided[i] {
            return Err(type_error(format!(
                "{name}() got multiple values for argument '{}'",
                names[i]
            )));
        }
        provided[i] = true;
    }
    if args.len() > names.len() {
        let count = if required == names.len() {
            format!("{required}")
        } else {
            format!("from {required} to {}", names.len())
        };
        return Err(type_error(format!(
            "{name}() takes {count} positional argument{} but {} were given",
            if required == 1 && names.len() == 1 {
                ""
            } else {
                "s"
            },
            args.len()
        )));
    }
    let missing = (0..required)
        .filter(|i| !provided[*i])
        .map(|i| format!("'{}'", names[i]))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        let description = match missing.as_slice() {
            [one] => one.clone(),
            [one, two] => format!("{one} and {two}"),
            many => format!(
                "{}, and {}",
                many[..many.len() - 1].join(", "),
                many.last().unwrap()
            ),
        };
        return Err(type_error(format!(
            "{name}() missing {} required positional argument{}: {}",
            missing.len(),
            if missing.len() == 1 { "" } else { "s" },
            description
        )));
    }
    Ok(())
}

/// UInt is a storage representation for Python int, not a distinct Python type.
/// Keep the original signed specialization whenever the value fits int64.
pub(crate) fn small_unsigned(value: &Value) -> bool {
    match value {
        Value::UInt(n) => *n <= i64::MAX as u64,
        Value::List(v) | Value::Tuple(v) => v.iter().any(small_unsigned),
        Value::Dict(v) => v.iter().any(|(_, v)| small_unsigned(v)),
        _ => false,
    }
}
pub(crate) fn python_integer_value(value: &Value) -> Value {
    match value {
        Value::UInt(n) if *n <= i64::MAX as u64 => Value::Int(*n as i64),
        Value::Tuple(v) => Value::Tuple(v.iter().map(python_integer_value).collect()),
        Value::List(v) => Value::List(v.iter().map(python_integer_value).collect()),
        Value::Dict(v) => Value::Dict(
            v.iter()
                .map(|(k, v)| (k.clone(), python_integer_value(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}
