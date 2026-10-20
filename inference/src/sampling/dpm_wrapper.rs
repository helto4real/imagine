//! The two nested functions created by ksampler, including their keyword
//! binding and source module lookup. Solver defaults belong to the callee.
use super::factory::DpmWrapper;
use crate::{execution_value::bind_arguments, native::Error, patching::options::Options};

/// Adaptive's optional info return does not change the original short-sigma
/// early return: even return_info=True then returns just the input tensor.
pub enum AdaptiveResult {
    Samples(tch::Tensor),
    WithInfo(tch::Tensor, super::dpm::Info),
}
pub fn adaptive(
    exec: &mut dyn super::solvers::Execution,
    native: &crate::native::Kitchen,
    x: tch::Tensor,
    sigmas: &tch::Tensor,
    settings: super::dpm::Adaptive,
    return_info: bool,
) -> Result<AdaptiveResult, Error> {
    let Some((minimum, maximum, _)) = super::dpm::bounds(sigmas)? else {
        return Ok(AdaptiveResult::Samples(x));
    };
    let (output, info) = super::dpm::adaptive(exec, native, x, &minimum, &maximum, settings)?;
    Ok(if return_info {
        AdaptiveResult::WithInfo(output, info)
    } else {
        AdaptiveResult::Samples(output)
    })
}

pub trait Bindings {
    fn length(&self, sigmas: &Options) -> Result<i64, Error>;
    fn index(&self, sigmas: &Options, index: i64) -> Result<Options, Error>;
    fn equals_zero(&self, sigma: &Options) -> Result<bool, Error>;
    fn module_attribute(&self, name: &str) -> Result<Options, Error>;
    fn call(
        &self,
        function: &Options,
        args: &[Options],
        kwargs: &[(String, Options)],
    ) -> Result<Options, Error>;
}

impl DpmWrapper {
    /// kwargs are the already-expanded keyword entries of a native call.
    /// Expansion itself remains the caller's dictionary/mapping protocol.
    pub fn call(
        self,
        args: &[Options],
        kwargs: &[(String, Options)],
        bindings: &dyn Bindings,
    ) -> Result<Options, Error> {
        const NAMES: [&str; 6] = [
            "model",
            "noise",
            "sigmas",
            "extra_args",
            "callback",
            "disable",
        ];
        let qualified = format!("ksampler.<locals>.{}", self.name());
        let mut fixed_keywords = vec![];
        let mut extra = vec![];
        for (name, value) in kwargs {
            if self == Self::Adaptive && !NAMES.contains(&name.as_str()) {
                extra.push((name.clone(), value.clone()));
            } else {
                fixed_keywords.push((name.clone(), value.clone()));
            }
        }
        bind_arguments(&qualified, &NAMES, NAMES.len(), args, &fixed_keywords)?;
        let fixed: Vec<Options> = NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| {
                args.get(i).cloned().unwrap_or_else(|| {
                    fixed_keywords
                        .iter()
                        .find(|(key, _)| key == name)
                        .unwrap()
                        .1
                        .clone()
                })
            })
            .collect();
        let sigmas = &fixed[2];
        if bindings.length(sigmas)? <= 1 {
            return Ok(fixed[1].clone());
        }
        let mut minimum = bindings.index(sigmas, -1)?;
        if bindings.equals_zero(&minimum)? {
            drop(std::mem::replace(&mut minimum, bindings.index(sigmas, -2)?));
        }
        let steps = if self == Self::Fast {
            Some(bindings.length(sigmas)? - 1)
        } else {
            None
        };
        // Resolve the actual module attribute before evaluating sigmas[0].
        let function = bindings.module_attribute(match self {
            Self::Fast => "sample_dpm_fast",
            Self::Adaptive => "sample_dpm_adaptive",
        })?;
        let maximum = bindings.index(sigmas, 0)?;
        let mut forwarded = vec![fixed[0].clone(), fixed[1].clone(), minimum.clone(), maximum];
        if let Some(steps) = steps {
            forwarded.push(Options::Atom(crate::execution_value::Value::Int(steps)));
        }
        let mut keywords = vec![
            ("extra_args".into(), fixed[3].clone()),
            ("callback".into(), fixed[4].clone()),
            ("disable".into(), fixed[5].clone()),
        ];
        keywords.extend(extra);
        bindings.call(&function, &forwarded, &keywords)
    }
}
