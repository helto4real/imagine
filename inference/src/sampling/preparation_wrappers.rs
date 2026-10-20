//! Source PREPARE_SAMPLING wrapper entry point over shared request objects.
use super::preparation::{Preparation, estimate_memory, prepare_with_estimate};
use crate::{
    conditioning_preprocess::Dimensions,
    execution_value::Value as V,
    native::Error,
    patching::{
        extensions::{Arguments, Original, Wrapper, WrapperExecutor},
        options::Options,
    },
};
use std::{cell::RefCell, rc::Rc};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn empty() -> Options {
    Options::from_value(V::Dict(vec![]))
}
fn values(value: &Options) -> Result<Vec<Options>, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().clone()),
        Options::Tuple(v) => Ok(v.as_ref().clone()),
        Options::Dict(v) => Ok(v
            .borrow()
            .iter()
            .map(|(k, _)| Options::Atom(V::Text(k.clone())))
            .collect()),
        Options::Atom(V::Text(v)) => Ok(v
            .chars()
            .map(|c| Options::Atom(V::Text(c.to_string())))
            .collect()),
        _ => Err(error(
            "TypeError",
            format!("'{}' object is not iterable", value.typename()),
        )),
    }
}
pub fn all_wrappers(options: &Options) -> Result<Vec<Wrapper<Options>>, Error> {
    all_wrappers_of("prepare_sampling", options)
}
pub fn all_wrappers_of(kind: &str, options: &Options) -> Result<Vec<Wrapper<Options>>, Error> {
    let transformer = options.get("transformer_options")?.unwrap_or_else(empty);
    let wrappers = transformer.get("wrappers")?.unwrap_or_else(empty);
    let registry = wrappers.get(kind)?.unwrap_or_else(empty);
    let mut output = vec![];
    let groups = registry.values_iterator()?;
    while let Some(group) = groups.next_value()? {
        let mut values = crate::patching::options_iterator::SequenceIterator::for_operation(
            &group,
            "sampler wrapper group",
        )?;
        while let Some(value) = values.next()? {
            if let Options::Object(object) = &value
                && let Some(wrapper) = object.downcast_ref::<Wrapper<Options>>()
            {
                output.push(wrapper.clone());
                continue;
            }
            // Source collection does not validate callability. Preserve failure
            // at invocation so earlier wrappers can still run or short-circuit.
            output.push(Rc::new(
                move |_: &WrapperExecutor<Options>, _: &[Options], _: &Arguments<Options>| {
                    Err(error(
                        "TypeError",
                        format!("'{}' object is not callable", value.typename()),
                    ))
                },
            ));
        }
    }
    Ok(output)
}

fn bind(args: &[Options], kwargs: &Arguments<Options>) -> Result<Vec<Options>, Error> {
    let names = [
        "model",
        "noise_shape",
        "conds",
        "model_options",
        "force_full_load",
        "force_offload",
    ];
    let mut values = vec![None; 6];
    for (slot, arg) in values.iter_mut().zip(args) {
        *slot = Some(arg.clone());
    }
    for (name, value) in kwargs {
        let Some(index) = names.iter().position(|n| n == name) else {
            return Err(error(
                "TypeError",
                format!("_prepare_sampling() got an unexpected keyword argument '{name}'"),
            ));
        };
        if values[index].is_some() {
            return Err(error(
                "TypeError",
                format!("_prepare_sampling() got multiple values for argument '{name}'"),
            ));
        }
        values[index] = Some(value.clone());
    }
    if args.len() > 6 {
        return Err(error(
            "TypeError",
            format!(
                "_prepare_sampling() takes from 3 to 6 positional arguments but {} were given",
                args.len()
            ),
        ));
    }
    let missing = (0..3)
        .filter(|i| values[*i].is_none())
        .map(|i| format!("'{}'", names[i]))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        let list = match missing.len() {
            1 => missing[0].clone(),
            2 => format!("{} and {}", missing[0], missing[1]),
            _ => format!("{}, {}, and {}", missing[0], missing[1], missing[2]),
        };
        return Err(error(
            "TypeError",
            format!(
                "_prepare_sampling() missing {} required positional argument{}: {list}",
                missing.len(),
                if missing.len() == 1 { "" } else { "s" }
            ),
        ));
    }
    Ok(values
        .into_iter()
        .enumerate()
        .map(|(i, value)| {
            value.unwrap_or_else(|| Options::Atom(if i == 3 { V::None } else { V::Bool(false) }))
        })
        .collect())
}
fn dimensions(value: &Options) -> Result<Vec<i64>, Error> {
    values(value)?
        .into_iter()
        .map(|v| match v {
            Options::Atom(V::Int(v)) => Ok(v),
            Options::Atom(V::Bool(v)) => Ok(i64::from(v)),
            Options::Atom(V::UInt(v)) => i64::try_from(v).map_err(|_| Error::BindingGap {
                backend: "prepare_sampling".into(),
                operation: "noise dimension exceeds i64".into(),
            }),
            _ => Err(Error::BindingGap {
                backend: "prepare_sampling".into(),
                operation: "non-native noise shape dimension".into(),
            }),
        })
        .collect()
}
pub struct Request {
    pub model: Options,
    pub noise_shape: Options,
    pub conditions: Options,
    pub model_options: Options,
    pub force_full_load: bool,
    pub force_offload: bool,
}
pub fn prepare(access: Rc<RefCell<dyn Preparation>>, request: Request) -> Result<Options, Error> {
    let wrappers = all_wrappers(&request.model_options)?;
    let body: Original<Options> = Rc::new(move |args, kwargs| {
        let values = bind(args, kwargs)?;
        let out = prepare_with_estimate(
            &mut *access.borrow_mut(),
            &values[0],
            &values[2],
            Some(&values[3]),
            (&values[4], &values[5]),
            |access, conditions| {
                let shape = dimensions(&values[1])?;
                let shape = if matches!(&values[1], Options::List(_)) {
                    Dimensions::List(&shape)
                } else {
                    Dimensions::Tuple(&shape)
                };
                estimate_memory(access, shape, conditions)
            },
        )?;
        Ok(Options::Tuple(Rc::new(vec![
            out.real_model,
            out.conditions,
            Options::List(Rc::new(RefCell::new(out.models))),
        ])))
    });
    let executor = WrapperExecutor::new(body, None, &wrappers, 0);
    drop(wrappers);
    executor.execute(
        &[request.model, request.noise_shape, request.conditions],
        &vec![
            ("model_options".into(), request.model_options),
            (
                "force_full_load".into(),
                Options::Atom(V::Bool(request.force_full_load)),
            ),
            (
                "force_offload".into(),
                Options::Atom(V::Bool(request.force_offload)),
            ),
        ],
    )
}
