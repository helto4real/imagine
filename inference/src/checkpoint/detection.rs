//! Shared shape/key-based detection helpers from the locked model_detection.py.
use super::state_dict;
use crate::{execution_value::Value as V, native::Error, patching::options::Options};
use std::{cell::RefCell, rc::Rc};
/// The complete frozen detector control flow, compiled into Rust at build time.
/// Registry admission requires differential evidence for each family/variant;
/// unsupported metadata/value operations remain explicit BindingGap errors.
pub fn detect_unet_config(
    runtime: &crate::execution::Runtime,
    state: &Options,
    key_prefix: &str,
    metadata: Option<&Options>,
) -> Result<Options, Error> {
    super::detector_program::run(super::detector_values::Context::new(
        runtime,
        state.clone(),
        key_prefix,
        metadata.cloned().unwrap_or(Options::Atom(V::None)),
    ))
}
/// Original Diffusers UNet header/config conversion, including the source's
/// ordered 18-variant matching table. This does not construct a model.
pub fn unet_config_from_diffusers_unet(
    runtime: &crate::execution::Runtime,
    state: &Options,
    dtype: Option<&Options>,
) -> Result<Options, Error> {
    let mut context =
        super::detector_values::Context::new(runtime, state.clone(), "", Options::Atom(V::None));
    context.set("dtype", dtype.cloned().unwrap_or(Options::Atom(V::None)));
    super::diffusers_program::run(context)
}
fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
/// The caller supplies the source prefix's formatting operation; every frozen
/// detector call uses a Rust format closure rather than a Python format runtime.
pub fn count_blocks(keys: &[String], prefix: impl Fn(usize) -> String) -> usize {
    let mut count = 0;
    while keys.iter().any(|key| key.starts_with(&prefix(count))) {
        count += 1;
    }
    count
}
pub fn any_suffix_in(keys: &[String], prefix: &str, main: &str, suffixes: &[String]) -> bool {
    suffixes.iter().any(|suffix| {
        keys.iter()
            .any(|key| key == &format!("{prefix}{main}{suffix}"))
    })
}
pub struct TransformerDepth {
    pub depth: usize,
    pub context_dimension: i64,
    pub linear: bool,
    pub time_stack: bool,
    pub time_stack_cross: bool,
}
fn required(sd: &Options, key: &str) -> Result<Options, Error> {
    sd.get(key)?.ok_or_else(|| {
        exception(
            "KeyError",
            crate::triton_specialization::python_str_repr(key),
        )
    })
}
fn shape(value: &Options) -> Result<Vec<i64>, Error> {
    match value {
        Options::Atom(V::Tensor(v)) => Ok(v.lock().unwrap().size()),
        Options::Atom(V::Quantized(v)) => Ok(v.lock().unwrap().params.original_shape.clone()),
        _ => Err(exception(
            "AttributeError",
            format!("'{}' object has no attribute 'shape'", value.typename()),
        )),
    }
}
pub fn calculate_transformer_depth(
    prefix: &str,
    keys: &[String],
    sd: &Options,
) -> Result<Option<TransformerDepth>, Error> {
    let transformer = format!("{prefix}1.transformer_blocks.");
    if !keys.iter().any(|key| key.starts_with(&transformer)) {
        return Ok(None);
    }
    let depth = count_blocks(keys, |n| format!("{transformer}{n}"));
    let dims = shape(&required(sd, &format!("{transformer}0.attn2.to_k.weight"))?)?;
    let context_dimension = *dims
        .get(1)
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    let linear = shape(&required(sd, &format!("{prefix}1.proj_in.weight"))?)?.len() == 2;
    let has = |suffix: &str| sd.get(&format!("{prefix}1.{suffix}")).map(|v| v.is_some());
    let time_stack =
        has("time_stack.0.attn1.to_q.weight")? || has("time_mix_blocks.0.attn1.to_q.weight")?;
    let time_stack_cross =
        has("time_stack.0.attn2.to_q.weight")? || has("time_mix_blocks.0.attn2.to_q.weight")?;
    Ok(Some(TransformerDepth {
        depth,
        context_dimension,
        linear,
        time_stack,
        time_stack_cross,
    }))
}
fn is_integer(value: &Options) -> bool {
    matches!(value, Options::Atom(V::Bool(_) | V::Int(_) | V::UInt(_)))
}
fn integer(value: &Options) -> Result<i64, Error> {
    match value {
        Options::Atom(V::Bool(v)) => Ok(i64::from(*v)),
        Options::Atom(V::Int(v)) => Ok(*v),
        Options::Atom(V::UInt(v)) => i64::try_from(*v).map_err(|_| Error::BindingGap {
            backend: "model_detection".into(),
            operation: "configuration repeat count exceeds i64".into(),
        }),
        _ => Err(exception(
            "TypeError",
            format!(
                "can't multiply sequence by non-int of type '{}'",
                value.typename()
            ),
        )),
    }
}
fn len(value: &Options) -> Result<usize, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().len()),
        Options::Tuple(v) => Ok(v.len()),
        Options::Dict(v) => Ok(v.borrow().len()),
        Options::Atom(V::Text(v)) => Ok(v.chars().count()),
        _ => Err(exception(
            "TypeError",
            format!("object of type '{}' has no len()", value.typename()),
        )),
    }
}
fn index(value: &Options, n: i64) -> Result<Options, Error> {
    let values = match value {
        Options::List(v) => v.borrow().clone(),
        Options::Tuple(v) => v.as_ref().clone(),
        _ => {
            return Err(exception(
                "TypeError",
                format!("'{}' object is not subscriptable", value.typename()),
            ));
        }
    };
    let n = if n < 0 { n + values.len() as i64 } else { n };
    values.get(n as usize).cloned().ok_or_else(|| {
        exception(
            "IndexError",
            if matches!(value, Options::Tuple(_)) {
                "tuple index out of range"
            } else {
                "list index out of range"
            },
        )
    })
}
fn repeat(value: &Options, n: i64) -> Options {
    Options::List(Rc::new(RefCell::new(vec![
        value.clone();
        n.max(0) as usize
    ])))
}
fn contains(sequence: &Options, number: i64) -> Result<bool, Error> {
    let values = match sequence {
        Options::List(v) => v.borrow().clone(),
        Options::Tuple(v) => v.as_ref().clone(),
        Options::Dict(_) => return Ok(false),
        Options::Atom(V::Text(_)) => {
            return Err(exception(
                "TypeError",
                "'in <string>' requires string as left operand, not int",
            ));
        }
        _ => {
            return Err(exception(
                "TypeError",
                format!("argument of type '{}' is not iterable", sequence.typename()),
            ));
        }
    };
    for value in values {
        if crate::conditioning::is_equal(&value.snapshot()?, &V::Int(number))? {
            return Ok(true);
        }
    }
    Ok(false)
}
fn absent(value: Option<Options>) -> Options {
    value.unwrap_or(Options::Atom(V::None))
}
pub fn convert_config(config: &Options) -> Result<Options, Error> {
    let out = config.shallow_copy_dict()?;
    let mut res = absent(out.get("num_res_blocks")?);
    let channels = absent(out.get("channel_mult")?);
    if is_integer(&res) {
        res = repeat(&res, len(&channels)? as i64);
    }
    if let Some(attention) = out.pop("attention_resolutions")? {
        let mut depth = absent(out.get("transformer_depth")?);
        let mut middle = absent(out.get("transformer_depth_middle")?);
        if is_integer(&depth) {
            depth = repeat(&depth, len(&channels)? as i64);
        }
        if matches!(middle, Options::Atom(V::None)) {
            middle = index(&depth, -1)?;
        }
        let mut input = vec![];
        let mut output = vec![];
        let mut scale = 1_i64;
        for i in 0..len(&res)? {
            let blocks = index(&res, i as i64)?;
            let d = if contains(&attention, scale)? {
                index(&depth, i as i64)?
            } else {
                Options::Atom(V::Int(0))
            };
            input.extend(children(&repeat(&d, integer(&blocks)?)));
            let next = match blocks {
                Options::Atom(V::Bool(v)) => i64::from(v) + 1,
                Options::Atom(V::Int(v)) => v.checked_add(1).ok_or_else(|| Error::BindingGap {
                    backend: "model_detection".into(),
                    operation: "configuration block count exceeds i64".into(),
                })?,
                Options::Atom(V::UInt(v)) => i64::try_from(v)
                    .ok()
                    .and_then(|v| v.checked_add(1))
                    .ok_or_else(|| Error::BindingGap {
                    backend: "model_detection".into(),
                    operation: "configuration block count exceeds i64".into(),
                })?,
                _ => unreachable!("sequence repeat validated an integer"),
            };
            output.extend(children(&repeat(&d, next)));
            scale = scale.checked_mul(2).ok_or_else(|| Error::BindingGap {
                backend: "model_detection".into(),
                operation: "configuration resolution scale exceeds i64".into(),
            })?;
        }
        out.set(
            "transformer_depth",
            Options::List(Rc::new(RefCell::new(input))),
        )?;
        out.set(
            "transformer_depth_output",
            Options::List(Rc::new(RefCell::new(output))),
        )?;
        out.set("transformer_depth_middle", middle)?;
    }
    out.set("num_res_blocks", res)?;
    Ok(out)
}
fn children(value: &Options) -> Vec<Options> {
    let Options::List(values) = value else {
        unreachable!()
    };
    values.borrow().clone()
}

pub fn prefix(sd: &Options) -> Result<&'static str, Error> {
    state_dict::unet_prefix(sd)
}
