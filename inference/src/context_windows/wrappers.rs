//! Original context sampling wrappers, preserving forwarding and lookup order.
use super::{Handler, error, gap};
use crate::{
    conditioning,
    execution_value::Value as V,
    native::Error,
    patching::{
        extensions::{Arguments, Wrapper},
        options::Options,
    },
};
use std::{cell::RefCell, rc::Rc};
fn values(v: &Options) -> Result<Vec<Options>, Error> {
    match v {
        Options::List(v) => Ok(v.borrow().clone()),
        Options::Tuple(v) => Ok((**v).clone()),
        Options::Dict(v) => Ok(v
            .borrow()
            .iter()
            .map(|(k, _)| Options::Atom(V::Text(k.clone())))
            .collect()),
        Options::Atom(V::Text(v)) => Ok(v
            .chars()
            .map(|v| Options::Atom(V::Text(v.to_string())))
            .collect()),
        _ => Err(error(
            "TypeError",
            format!("'{}' object is not iterable", v.typename()),
        )),
    }
}
fn index(dim: i64, len: usize) -> Result<usize, Error> {
    let index = if dim < 0 {
        len as i128 + i128::from(dim)
    } else {
        i128::from(dim)
    };
    usize::try_from(index)
        .ok()
        .filter(|i| *i < len)
        .ok_or_else(|| error("IndexError", "list index out of range"))
}
fn greater(value: &Options, right: i64) -> Result<bool, Error> {
    Ok(match value {
        Options::Atom(V::Bool(v)) => i64::from(*v) > right,
        Options::Atom(V::Int(v)) => *v > right,
        Options::Atom(V::UInt(v)) => i128::from(*v) > i128::from(right),
        // Integral dimension values do not pass through floating point.
        Options::Atom(V::Float(v)) if v.is_nan() => false,
        Options::Atom(V::Float(v)) => {
            if *v >= 9223372036854775808. {
                true
            } else if *v < -9223372036854775808. {
                false
            } else {
                let whole = v.trunc() as i64;
                whole > right || (whole == right && v.fract() > 0.)
            }
        }
        Options::Object(_) | Options::Atom(V::Tensor(_) | V::Quantized(_)) => {
            return Err(gap(
                "context shape comparison requires its original object binding",
            ));
        }
        _ => {
            return Err(error(
                "TypeError",
                format!(
                    "'>' not supported between instances of '{}' and 'int'",
                    value.typename()
                ),
            ));
        }
    })
}
fn same_one(value: &Options) -> Result<bool, Error> {
    if matches!(value, Options::Object(_)) {
        return Err(gap(
            "context packed-shape equality requires its original object binding",
        ));
    }
    conditioning::is_equal(&value.snapshot()?, &V::Int(1))
}
/// Returns the original shape when no handler is installed. With a handler,
/// even a packed/no-clamp shape is copied to the source's fresh list.
pub fn prepare_sampling_shape(shape: &Options, model_options: &Options) -> Result<Options, Error> {
    let handler = model_options
        .get("context_handler")?
        .unwrap_or(Options::Atom(V::None));
    if matches!(handler, Options::Atom(V::None)) {
        return Ok(shape.clone());
    }
    let mut shape = values(shape)?;
    let packed = shape.len() == 3 && same_one(&shape[1])?;
    if !packed {
        let h = match &handler {
            Options::Object(v) => v.downcast_ref::<Handler>(),
            _ => None,
        }
        .ok_or_else(|| {
            if matches!(handler, Options::Object(_)) {
                gap("context prepare wrapper requires the handler's original attribute binding")
            } else {
                error(
                    "AttributeError",
                    format!("'{}' object has no attribute 'dim'", handler.typename()),
                )
            }
        })?;
        let dim = h.settings.borrow().dim;
        if i128::from(dim) < shape.len() as i128 {
            let at = index(dim, shape.len())?;
            let length = h.settings.borrow().length;
            if greater(&shape[at], length)? {
                shape[at] = Options::Atom(V::Int(length));
            }
        }
    }
    Ok(Options::List(Rc::new(RefCell::new(shape))))
}
fn bind(
    function: &str,
    names: &[&str],
    args: &[Options],
    kwargs: &Arguments<Options>,
) -> Result<(Vec<Options>, Arguments<Options>), Error> {
    let mut fixed: Vec<_> = (0..names.len()).map(|i| args.get(i).cloned()).collect();
    let mut rest = vec![];
    for (name, value) in kwargs {
        if name == "executor" {
            return Err(error(
                "TypeError",
                format!("{function}() got multiple values for argument 'executor'"),
            ));
        }
        if let Some(i) = names.iter().position(|v| v == name) {
            if fixed[i].is_some() {
                return Err(error(
                    "TypeError",
                    format!("{function}() got multiple values for argument '{name}'"),
                ));
            }
            fixed[i] = Some(value.clone());
        } else {
            rest.push((name.clone(), value.clone()));
        }
    }
    let missing: Vec<_> = fixed
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_none())
        .map(|(i, _)| format!("'{}'", names[i]))
        .collect();
    if !missing.is_empty() {
        let items = match missing.len() {
            1 => missing[0].clone(),
            2 => format!("{} and {}", missing[0], missing[1]),
            _ => format!(
                "{}, and {}",
                missing[..missing.len() - 1].join(", "),
                missing.last().unwrap()
            ),
        };
        return Err(error(
            "TypeError",
            format!(
                "{function}() missing {} required positional argument{}: {items}",
                missing.len(),
                if missing.len() == 1 { "" } else { "s" }
            ),
        ));
    }
    let mut fixed: Vec<_> = fixed.into_iter().map(Option::unwrap).collect();
    fixed.extend(args.iter().skip(names.len()).cloned());
    Ok((fixed, rest))
}
pub fn prepare_sampling_wrapper() -> Wrapper<Options> {
    Rc::new(|executor, args, kwargs| {
        let (mut args, kwargs) = bind(
            "_prepare_sampling_wrapper",
            &["model", "noise_shape", "conds"],
            args,
            kwargs,
        )?;
        let model_options=kwargs.iter().find(|(k,_)|k=="model_options").map(|(_,v)|v).filter(|v|!matches!(v,Options::Atom(V::None))).ok_or_else(||error("Exception","model_options not found in prepare_sampling_wrapper; this should never happen, something went wrong."))?;
        args[1] = prepare_sampling_shape(&args[1], model_options)?;
        executor.next(&args, &kwargs)
    })
}

pub trait SamplerWrapperBindings: super::ConditionAttributes {
    /// Each call is a separate source attribute read, including descriptor
    /// effects. The returned value is the actual shared conditions mapping.
    fn guider_conditions(&self, guider: &Options) -> Result<Options, Error>;
    fn info(&self, message: &str) -> Result<(), Error>;
    fn freenoise_call(&self, _: &V, _: &Options, _: &Options) -> Result<(), Error> {
        Ok(())
    }
}
fn mapping_get(value: &Options, key: &str) -> Result<Option<Options>, Error> {
    if matches!(value, Options::Object(_)) {
        return Err(gap(
            "context sampler mapping requires its original get binding",
        ));
    }
    value.get(key)
}
pub fn sampler_sample_wrapper(
    native: crate::native::Kitchen,
    bindings: Rc<dyn SamplerWrapperBindings>,
) -> Wrapper<Options> {
    Rc::new(move |executor, args, kwargs| {
        let (mut args, kwargs) = bind(
            "_sampler_sample_wrapper",
            &["guider", "sigmas", "extra_args", "callback", "noise"],
            args,
            kwargs,
        )?;
        let extra = &args[2];
        let model_options = mapping_get(extra, "model_options")?.filter(|v| !matches!(v, Options::Atom(V::None)))
            .ok_or_else(|| error("Exception", "model_options not found in sampler_sample_wrapper; this should never happen, something went wrong."))?;
        let handler = mapping_get(&model_options, "context_handler")?.filter(|v| !matches!(v, Options::Atom(V::None)))
            .ok_or_else(|| error("Exception", "context_handler not found in sampler_sample_wrapper; this should never happen, something went wrong."))?;
        let h = match &handler {
            Options::Object(v) => v.downcast_ref::<Handler>(),
            _ => None,
        }
        .ok_or_else(|| {
            if matches!(handler, Options::Object(_)) {
                gap("context sampler wrapper requires the handler's original attribute binding")
            } else {
                error(
                    "AttributeError",
                    format!(
                        "'{}' object has no attribute 'freenoise'",
                        handler.typename()
                    ),
                )
            }
        })?;
        if !h.settings.borrow().freenoise {
            return executor.next(&args, &kwargs);
        }
        // The positive .get method is resolved first. Its default argument
        // then reads guider.conds again and calls negative .get eagerly.
        let positive_mapping = bindings.guider_conditions(&args[0])?;
        if !matches!(positive_mapping, Options::Dict(_)) {
            if matches!(positive_mapping, Options::Object(_)) {
                return Err(gap(
                    "context sampler mapping requires its original get binding",
                ));
            }
            return Err(error(
                "AttributeError",
                format!(
                    "'{}' object has no attribute 'get'",
                    positive_mapping.typename()
                ),
            ));
        }
        let negative_mapping = bindings.guider_conditions(&args[0])?;
        let negative = mapping_get(&negative_mapping, "negative")?
            .unwrap_or_else(|| Options::List(Rc::new(RefCell::new(vec![]))));
        drop(negative_mapping);
        let selected = positive_mapping.get("positive")?.unwrap_or(negative);
        drop(positive_mapping);
        let conds = Options::List(Rc::new(RefCell::new(vec![selected])));
        let noise = {
            let seed = super::evaluation::required(extra, "seed")?;
            let noise = args[4].snapshot()?;
            bindings.freenoise_call(&noise, &conds, &seed)?;
            h.apply_freenoise_value(
                &native,
                &noise,
                &conds,
                &seed.snapshot()?,
                bindings.as_ref(),
                &mut |message| bindings.info(message),
            )?
        };
        args[4] = Options::Atom(noise);
        executor.next(&args, &kwargs)
    })
}

pub trait ContextWrapperModel {
    fn add_wrapper_with_key(
        &self,
        kind: &str,
        key: &str,
        wrapper: Wrapper<Options>,
    ) -> Result<(), Error>;
}
/// Canonical function objects for one bound context module. Registering the
/// same helper repeatedly passes the same function owner, as the source does.
pub struct ContextWrappers {
    prepare: Wrapper<Options>,
    sampler: Wrapper<Options>,
}
impl ContextWrappers {
    pub fn new(native: crate::native::Kitchen, bindings: Rc<dyn SamplerWrapperBindings>) -> Self {
        Self {
            prepare: prepare_sampling_wrapper(),
            sampler: sampler_sample_wrapper(native, bindings),
        }
    }
    pub fn create_prepare_sampling_wrapper(
        &self,
        model: &dyn ContextWrapperModel,
    ) -> Result<(), Error> {
        model.add_wrapper_with_key(
            "prepare_sampling",
            "ContextWindows_prepare_sampling",
            self.prepare.clone(),
        )
    }
    pub fn create_sampler_sample_wrapper(
        &self,
        model: &dyn ContextWrapperModel,
    ) -> Result<(), Error> {
        model.add_wrapper_with_key(
            "sampler_sample",
            "ContextWindows_sampler_sample",
            self.sampler.clone(),
        )
    }
}
