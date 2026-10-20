//! Source bypass module metadata and inference-weight movement. This does not
//! merge weights or replace a module forward; hook ownership is managed above.
use crate::{
    execution::Runtime, execution_value::Value as V, native::Error, patching::options::Options,
};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Kind};
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum KnownType {
    Conv1d,
    Conv2d,
    Conv3d,
    Linear,
}
pub struct ModuleClass {
    pub name: String,
    pub instances: Vec<KnownType>,
}
fn atom(value: V) -> Options {
    Options::Atom(value)
}
fn tuple(value: Options, count: usize) -> Options {
    Options::Tuple(Rc::new(vec![value; count]))
}
thread_local! {
    // The source function's immutable tuple constants survive every call.
    static ONE:Options=tuple(atom(V::Int(1)),1);
    static ZERO:Options=tuple(atom(V::Int(0)),1);
}
fn default_tuple(value: i64, count: usize) -> Options {
    if count == 1 {
        if value == 1 {
            ONE.with(Clone::clone)
        } else {
            ZERO.with(Clone::clone)
        }
    } else {
        tuple(atom(V::Int(value)), count)
    }
}
pub fn get_module_type_info(class: &ModuleClass, attributes: &Options) -> Result<Options, Error> {
    let info = Options::dictionary(vec![
        ("is_conv".into(), atom(V::Bool(false))),
        ("conv_dim".into(), atom(V::Int(0))),
        ("stride".into(), default_tuple(1, 1)),
        ("padding".into(), default_tuple(0, 1)),
        ("dilation".into(), default_tuple(1, 1)),
        ("groups".into(), atom(V::Int(1))),
        ("kernel_size".into(), default_tuple(1, 1)),
        ("in_channels".into(), atom(V::None)),
        ("out_channels".into(), atom(V::None)),
    ]);
    let dimension = if class.instances.contains(&KnownType::Conv1d) {
        1
    } else if class.instances.contains(&KnownType::Conv2d) {
        2
    } else if class.instances.contains(&KnownType::Conv3d) {
        3
    } else if class.instances.contains(&KnownType::Linear) {
        0
    } else {
        let name = class.name.to_lowercase();
        if name.contains("conv3d") {
            3
        } else if name.contains("conv2d") {
            2
        } else if name.contains("conv1d") {
            1
        } else if name.contains("conv") {
            2
        } else {
            0
        }
    };
    if dimension > 0 {
        info.set("is_conv", atom(V::Bool(true)))?;
        info.set("conv_dim", atom(V::Int(dimension as i64)))?;
        // Preserve the source attribute-read and subsequent normalization order.
        for (name, default) in [
            ("stride", default_tuple(1, dimension)),
            ("padding", default_tuple(0, dimension)),
            ("dilation", default_tuple(1, dimension)),
            ("groups", atom(V::Int(1))),
            ("kernel_size", default_tuple(1, dimension)),
            ("in_channels", atom(V::None)),
            ("out_channels", atom(V::None)),
        ] {
            info.set(name, attributes.get(name)?.unwrap_or(default))?;
        }
        for name in ["stride", "padding", "dilation", "kernel_size"] {
            let value = info.get(name)?.unwrap();
            if matches!(value, Options::Atom(V::Int(_) | V::UInt(_) | V::Bool(_))) {
                info.set(name, tuple(value, dimension))?;
            }
        }
    }
    Ok(info)
}
pub fn constructor_attributes(
    adapter: &Options,
    info: &Options,
    multiplier: f64,
) -> Result<(), Error> {
    adapter.set("multiplier", atom(V::Float(multiplier)))?;
    for name in [
        "is_conv",
        "conv_dim",
        "kernel_size",
        "in_channels",
        "out_channels",
    ] {
        adapter.set(name, info.get(name)?.unwrap())?;
    }
    let kwargs = Options::dictionary(vec![]);
    if matches!(info.get("is_conv")?, Some(Options::Atom(V::Bool(true)))) {
        for name in ["stride", "padding", "dilation", "groups"] {
            kwargs.set(name, info.get(name)?.unwrap())?;
        }
    }
    adapter.set("kw_dict", kwargs)
}
pub fn weight_dtype(weight: Option<&V>) -> Result<Option<Kind>, Error> {
    let kind = match weight {
        None | Some(V::None) => return Ok(None),
        Some(V::Tensor(_)) => weight.unwrap().as_tensor()?.kind(),
        Some(V::Quantized(value)) => value.lock().unwrap().params.original_kind,
        Some(value) => {
            return Err(Error::ReferenceException {
                class: "AttributeError",
                message: format!(
                    "'{}' object has no attribute 'dtype'",
                    Options::from_value(value.clone()).typename()
                ),
            });
        }
    };
    Ok(matches!(kind, Kind::Float | Kind::Half | Kind::BFloat16).then_some(kind))
}
/// # Safety
/// The runtime must use the locked ABI and every tensor/quantized storage must
/// satisfy its original device, stream and operand-capacity requirements.
pub unsafe fn move_adapter_weights(
    rt: &Runtime,
    adapter: &Options,
    device: Device,
    dtype: Option<Kind>,
) -> Result<(), Error> {
    let Some(weights) = adapter
        .get("weights")?
        .filter(|v| !matches!(v, Options::Atom(V::None)))
    else {
        return Ok(());
    };
    let convert = |value: &Options| -> Result<Options, Error> {
        let Options::Atom(value @ (V::Tensor(_) | V::Quantized(_))) = value else {
            return Ok(value.clone());
        };
        let kwargs = vec![
            ("device".into(), V::Device(device)),
            ("dtype".into(), dtype.map_or(V::None, V::DType)),
        ];
        Ok(atom(unsafe {
            rt.op(
                "aten::to.dtype_layout",
                std::slice::from_ref(value),
                &kwargs,
            )
        }?))
    };
    match weights {
        Options::List(values) => {
            let mut output = Vec::new();
            for value in values.borrow().iter() {
                output.push(convert(value)?);
            }
            adapter.set("weights", Options::List(Rc::new(RefCell::new(output))))?;
        }
        Options::Tuple(values) => {
            if values.is_empty() {
                adapter.set("weights", Options::Tuple(values))?;
                return Ok(());
            }
            let mut output = Vec::new();
            for value in values.iter() {
                output.push(convert(value)?);
            }
            adapter.set("weights", Options::Tuple(Rc::new(output)))?;
        }
        Options::Atom(V::Tensor(_) | V::Quantized(_)) => {
            adapter.set("weights", convert(&weights)?)?
        }
        _ => {}
    }
    Ok(())
}
