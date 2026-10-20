//! Source sampler_helpers dependency collection and memory-shape estimation.
//! Model/controller operations remain required typed Rust bindings.
use crate::{
    conditioning_preprocess::Dimensions,
    execution_value::Value as V,
    native::Error,
    patching::{
        hooks::{HookGroup, Kind},
        options::Options,
    },
};
use std::{cell::RefCell, rc::Rc};
use tch::Kind as Dtype;

fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn empty_dict() -> Options {
    Options::dictionary(vec![])
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn sequence(value: &Options) -> Result<Vec<Options>, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().clone()),
        Options::Tuple(v) => Ok(v.as_ref().clone()),
        _ => Err(error(
            "TypeError",
            format!("'{}' object is not iterable", value.typename()),
        )),
    }
}
/// Shape values keep their original tuple/list objects in both shape maps.
/// Products compare exactly in the ordinary ATen integer-shape domain.
fn product(value: &Options) -> Result<i128, Error> {
    let mut output = 1_i128;
    for n in sequence(value)? {
        let number = match n.snapshot()? {
            V::Int(x) => x as i128,
            V::UInt(x) => x as i128,
            V::Bool(x) => i128::from(x),
            _ => {
                return Err(Error::BindingGap {
                    backend: "sampling memory".into(),
                    operation: "non-integer model shape product".into(),
                });
            }
        };
        output = output
            .checked_mul(number)
            .ok_or_else(|| Error::BindingGap {
                backend: "sampling memory".into(),
                operation: "shape product beyond i128".into(),
            })?;
    }
    Ok(output)
}
pub trait MemoryEstimator {
    fn extra_conds_shapes(&mut self, record: &Options) -> Result<Options, Error>;
    fn memory_required(&mut self, shape: &[i64], cond_shapes: &Options) -> Result<f64, Error>;
}
pub fn estimate_memory(
    model: &mut dyn MemoryEstimator,
    noise_shape: Dimensions<'_>,
    conditions: &Options,
) -> Result<(f64, f64), Error> {
    let shapes = empty_dict();
    let minimum = empty_dict();
    for (_, records) in conditions.dict_items()? {
        for record in sequence(&records)? {
            for (name, value) in model.extra_conds_shapes(&record)?.dict_items()? {
                let current = match shapes.get(&name)? {
                    Some(v) => v,
                    None => {
                        let v = list(vec![]);
                        shapes.set(&name, v.clone())?;
                        v
                    }
                };
                let Options::List(values) = current else {
                    unreachable!("source defaultdict(list)")
                };
                values.borrow_mut().push(value.clone());
                if let Some(previous) = minimum
                    .get(&name)?
                    .filter(|v| !matches!(v, Options::Atom(V::None)))
                {
                    let values = sequence(&previous)?;
                    if product(&value)? > product(&values[0])? {
                        minimum.set(&name, list(vec![value]))?;
                    }
                } else {
                    minimum.set(&name, list(vec![value]))?;
                }
            }
        }
    }
    let (&batch, rest) = noise_shape.values().split_first().ok_or_else(|| {
        error(
            "IndexError",
            if matches!(noise_shape, Dimensions::List(_)) {
                "list index out of range"
            } else {
                "tuple index out of range"
            },
        )
    })?;
    let mut shape = vec![batch * 2];
    shape.extend_from_slice(rest);
    let required = model.memory_required(&shape, &shapes)?;
    let mut shape = vec![batch];
    shape.extend_from_slice(rest);
    let minimum = model.memory_required(&shape, &minimum)?;
    Ok((required, minimum))
}
pub fn get_models_from_cond(records: &[Options], name: &str) -> Result<Vec<Options>, Error> {
    let mut models = Vec::new();
    for record in records {
        if let Some(value) = record.get(name)? {
            if let Options::List(values) = value {
                models.extend(values.borrow().iter().cloned());
            } else {
                models.push(value);
            }
        }
    }
    Ok(models)
}
pub trait Dependencies {
    fn control_models(&mut self, control: &Options) -> Result<Vec<Options>, Error>;
    fn inference_memory_requirements(
        &mut self,
        control: &Options,
        dtype: Dtype,
    ) -> Result<f64, Error>;
    fn cleanup_if_present(&mut self, model: &Options) -> Result<(), Error>;
}
pub(super) fn truth(value: &Options) -> Result<bool, Error> {
    if let Some(value) = value.metadata_truth() {
        return value;
    }
    Ok(match value {
        Options::Dict(v) => !v.borrow().is_empty(),
        Options::List(v) => !v.borrow().is_empty(),
        Options::Tuple(v) => !v.is_empty(),
        Options::Object(_) => true,
        Options::Atom(V::None) => false,
        Options::Atom(V::Bool(v)) => *v,
        Options::Atom(V::Int(v)) => *v != 0,
        Options::Atom(V::UInt(v)) => *v != 0,
        Options::Atom(V::Float(v)) => *v != 0.,
        Options::Atom(V::Text(v)) => !v.is_empty(),
        Options::Atom(V::Tensor(_)) => value.snapshot()?.as_tensor()?.f_is_nonzero()?,
        Options::Atom(V::Quantized(_)) => {
            return Err(Error::BindingGap {
                backend: "prepare_sampling".into(),
                operation: "quantized force flag requires its original is_nonzero binding".into(),
            });
        }
        _ => true,
    })
}
/// Required model/loading operations of the source `_prepare_sampling` body.
/// Implementations bind the shared memory manager and concrete model owners;
/// this interface provides no alternate loading or offload policy.
pub trait Preparation: MemoryEstimator + Dependencies {
    fn match_multigpu_clones(&mut self, model: &Options) -> Result<(), Error>;
    fn preprocess_multigpu_conds(
        &mut self,
        conditions: &Options,
        model: &Options,
        model_options: Option<&Options>,
    ) -> Result<(), Error>;
    fn model_dtype(&mut self, model: &Options) -> Result<Dtype, Error>;
    fn nested_additional_models(&mut self, model: &Options) -> Result<Vec<Options>, Error>;
    fn load_models_gpu(
        &mut self,
        models: &[Options],
        memory_required: f64,
        minimum_memory_required: Option<f64>,
        force_full_load: &Options,
    ) -> Result<(), Error>;
    fn real_model(&mut self, model: &Options) -> Result<Options, Error>;
}
pub struct PreparedSampling {
    pub real_model: Options,
    pub conditions: Options,
    pub models: Vec<Options>,
}
pub fn prepare_unwrapped(
    access: &mut dyn Preparation,
    model: &Options,
    noise_shape: Dimensions<'_>,
    conditions: &Options,
    model_options: Option<&Options>,
    force_full_load: bool,
    force_offload: bool,
) -> Result<PreparedSampling, Error> {
    prepare_with_estimate(
        access,
        model,
        conditions,
        model_options,
        (
            &Options::Atom(V::Bool(force_full_load)),
            &Options::Atom(V::Bool(force_offload)),
        ),
        |access, conditions| estimate_memory(access, noise_shape, conditions),
    )
}
pub(super) fn prepare_with_estimate(
    access: &mut dyn Preparation,
    model: &Options,
    conditions: &Options,
    model_options: Option<&Options>,
    flags: (&Options, &Options),
    estimate: impl FnOnce(&mut dyn Preparation, &Options) -> Result<(f64, f64), Error>,
) -> Result<PreparedSampling, Error> {
    let (force_full_load, force_offload) = flags;
    access.match_multigpu_clones(model)?;
    access.preprocess_multigpu_conds(conditions, model, model_options)?;
    let dtype = access.model_dtype(model)?;
    let (mut models, inference_memory) = get_additional_models(conditions, dtype, access)?;
    models.extend(get_additional_models_from_model_options(model_options)?);
    models.extend(access.nested_additional_models(model)?);
    let (required, minimum) = if truth(force_offload)? {
        (1e20, None)
    } else {
        let (required, minimum) = estimate(access, conditions)?;
        (
            required + inference_memory,
            Some(minimum + inference_memory),
        )
    };
    let mut to_load = vec![model.clone()];
    to_load.extend(models.iter().cloned());
    access.load_models_gpu(&to_load, required, minimum, force_full_load)?;
    let real_model = access.real_model(model)?;
    Ok(PreparedSampling {
        real_model,
        conditions: conditions.clone(),
        models,
    })
}
fn same_control(a: &Options, b: &Options) -> Result<bool, Error> {
    if matches!(
        a,
        Options::Object(_) | Options::Atom(V::Tensor(_) | V::Quantized(_))
    ) || matches!(
        b,
        Options::Object(_) | Options::Atom(V::Tensor(_) | V::Quantized(_))
    ) {
        Ok(a.same_object(b))
    } else {
        crate::conditioning::is_equal(&a.snapshot()?, &b.snapshot()?)
    }
}
fn hashable(value: &Options) -> Result<(), Error> {
    match value {
        Options::Dict(_) | Options::List(_) => Err(error(
            "TypeError",
            format!("unhashable type: '{}'", value.typename()),
        )),
        Options::Tuple(v) => {
            for value in v.iter() {
                hashable(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
fn gligen_model(value: &Options) -> Result<Options, Error> {
    match value {
        Options::Atom(V::Text(text)) => text
            .chars()
            .nth(1)
            .map(|c| Options::Atom(V::Text(c.to_string())))
            .ok_or_else(|| error("IndexError", "string index out of range")),
        Options::List(v) => v
            .borrow()
            .get(1)
            .cloned()
            .ok_or_else(|| error("IndexError", "list index out of range")),
        Options::Tuple(v) => v
            .get(1)
            .cloned()
            .ok_or_else(|| error("IndexError", "tuple index out of range")),
        _ => Err(error(
            "TypeError",
            format!("'{}' object is not subscriptable", value.typename()),
        )),
    }
}
pub fn get_additional_models(
    conditions: &Options,
    dtype: Dtype,
    access: &mut dyn Dependencies,
) -> Result<(Vec<Options>, f64), Error> {
    let mut controls = Vec::new();
    let mut gligen = Vec::new();
    let mut additional = Vec::new();
    for (_, records) in conditions.dict_items()? {
        let records = sequence(&records)?;
        controls.extend(get_models_from_cond(&records, "control")?);
        gligen.extend(get_models_from_cond(&records, "gligen")?);
        additional.extend(get_models_from_cond(&records, "additional_models")?);
    }
    // Current source dict.fromkeys preserves the first control reference/order.
    let mut unique = Vec::new();
    for control in controls {
        hashable(&control)?;
        let mut present = false;
        for old in &unique {
            if same_control(old, &control)? {
                present = true;
                break;
            }
        }
        if !present {
            unique.push(control);
        }
    }
    let mut memory = 0.;
    let mut models = Vec::new();
    for control in unique {
        models.extend(access.control_models(&control)?);
        memory += access.inference_memory_requirements(&control, dtype)?;
    }
    for value in gligen {
        models.push(gligen_model(&value)?);
    }
    models.extend(additional);
    Ok((models, memory))
}
pub fn get_additional_models_from_model_options(
    options: Option<&Options>,
) -> Result<Vec<Options>, Error> {
    let Some(options) = options.filter(|v| !matches!(v, Options::Atom(V::None))) else {
        return Ok(vec![]);
    };
    let Some(hooks) = options.get("registered_hooks")? else {
        return Ok(vec![]);
    };
    let Options::Object(ref object) = hooks else {
        return Err(error(
            "AttributeError",
            format!("'{}' object has no attribute 'get_type'", hooks.typename()),
        ));
    };
    let group = object.downcast_ref::<RefCell<HookGroup>>().ok_or_else(|| {
        error(
            "AttributeError",
            "'object' object has no attribute 'get_type'",
        )
    })?;
    let hooks = group.borrow().get_type(Kind::AdditionalModels);
    let mut models = Vec::new();
    // Source iterates its actual cached list, including mutations during use.
    let mut index = 0;
    while index < hooks.borrow().len() {
        let hook = hooks.borrow()[index].clone();
        index += 1;
        models.extend(sequence(&hook.borrow().values)?);
    }
    Ok(models)
}
pub fn cleanup_additional_models(
    models: &[Options],
    access: &mut dyn Dependencies,
) -> Result<(), Error> {
    for model in models {
        access.cleanup_if_present(model)?;
    }
    Ok(())
}
