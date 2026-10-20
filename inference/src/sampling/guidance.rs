//! Source CFG and DualCFG arithmetic/hook order, independent of model families.
use crate::{native::Error, patching::options::Options};
use std::{cell::RefCell, rc::Rc};
use tch::Tensor;
/// The source prediction list is mutable and can also be retained by callbacks.
/// Keep the actual shared container, rather than a snapshot of its tensors.
pub type Predictions = Rc<RefCell<Vec<Tensor>>>;
pub fn predictions(values: Vec<Tensor>) -> Predictions {
    Rc::new(RefCell::new(values))
}
fn prediction(values: &Predictions, index: usize) -> Result<Tensor, Error> {
    values
        .borrow()
        .get(index)
        .map(Tensor::shallow_clone)
        .ok_or_else(|| Error::ReferenceException {
            class: "IndexError",
            message: "list index out of range".into(),
        })
}
pub struct Context<'a, C> {
    pub x: &'a Tensor,
    pub timestep: &'a Tensor,
    pub scale: f64,
    pub options: &'a Options,
    pub cond: Option<&'a C>,
    pub uncond: Option<&'a C>,
}
pub struct CfgInput<'a, C> {
    pub context: Context<'a, C>,
    pub cond_denoised: &'a Tensor,
    pub uncond_denoised: &'a Tensor,
    /// EPS-like arguments exist only for the source sampler_cfg_function hook.
    pub cond_noise: Option<&'a Tensor>,
    pub uncond_noise: Option<&'a Tensor>,
}
pub trait Hooks<C> {
    /// Retain the original callback-list object for this iteration. Replacing
    /// the options key must not replace the active iterator's list.
    fn begin_post(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn has_custom_cfg(&self) -> bool {
        false
    }
    fn custom_cfg(&mut self, _args: CfgInput<'_, C>) -> Result<Tensor, Error> {
        Err(Error::BindingGap {
            backend: "sampler_cfg_function".into(),
            operation: "custom CFG callback".into(),
        })
    }
    fn post_count(&self) -> usize {
        0
    }
    fn post(
        &mut self,
        _index: usize,
        _args: CfgInput<'_, C>,
        _denoised: Tensor,
    ) -> Result<Tensor, Error> {
        Err(Error::BindingGap {
            backend: "sampler_post_cfg_function".into(),
            operation: "post-CFG callback".into(),
        })
    }
}
pub struct NoHooks;
impl<C> Hooks<C> for NoHooks {}
pub fn is_close_one(value: f64) -> bool {
    value == 1. || (value.is_finite() && (value - 1.).abs() <= 1e-9 * value.abs().max(1.))
}
pub fn cfg<C>(
    hooks: &mut dyn Hooks<C>,
    context: Context<'_, C>,
    cond: &Tensor,
    uncond: &Tensor,
) -> Result<Tensor, Error> {
    let mut retained_args: Option<Vec<Tensor>> = None;
    let mut result = if hooks.has_custom_cfg() {
        let cond_noise = context.x.f_sub(cond)?;
        let uncond_noise = context.x.f_sub(uncond)?;
        retained_args = Some(vec![
            cond_noise.shallow_clone(),
            uncond_noise.shallow_clone(),
        ]);
        let output = hooks.custom_cfg(CfgInput {
            context: Context {
                x: context.x,
                timestep: context.timestep,
                scale: context.scale,
                options: context.options,
                cond: context.cond,
                uncond: context.uncond,
            },
            cond_denoised: cond,
            uncond_denoised: uncond,
            cond_noise: Some(&cond_noise),
            uncond_noise: Some(&uncond_noise),
        })?;
        context.x.f_sub(&output)?
    } else {
        uncond.f_add(&cond.f_sub(uncond)?.f_mul_scalar(context.scale)?)?
    };
    hooks.begin_post()?;
    let mut index = 0;
    while index < hooks.post_count() {
        retained_args = Some(vec![result.shallow_clone()]);
        result = hooks.post(
            index,
            CfgInput {
                context: Context {
                    x: context.x,
                    timestep: context.timestep,
                    scale: context.scale,
                    options: context.options,
                    cond: context.cond,
                    uncond: context.uncond,
                },
                cond_denoised: cond,
                uncond_denoised: uncond,
                cond_noise: None,
                uncond_noise: None,
            },
            result,
        )?;
        index += 1;
    }
    drop(retained_args);
    Ok(result)
}
pub trait Evaluator<C>: Hooks<C> {
    fn begin_pre(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn batch(
        &mut self,
        conditions: &[Option<&C>],
        x: &Tensor,
        timestep: &Tensor,
        options: &Options,
    ) -> Result<Predictions, Error>;
    fn pre_count(&self) -> usize {
        0
    }
    fn pre(
        &mut self,
        _index: usize,
        _conditions: &[Option<&C>],
        _context: &Context<'_, C>,
        _out: Predictions,
    ) -> Result<Predictions, Error> {
        Err(Error::BindingGap {
            backend: "sampler_pre_cfg_function".into(),
            operation: "pre-CFG callback".into(),
        })
    }
}
pub fn sampling<C>(
    execution: &mut dyn Evaluator<C>,
    mut context: Context<'_, C>,
    disable_cfg1: bool,
) -> Result<Tensor, Error> {
    if is_close_one(context.scale) && !disable_cfg1 {
        context.uncond = None;
    }
    let conditions = [context.cond, context.uncond];
    let mut out = execution.batch(&conditions, context.x, context.timestep, context.options)?;
    execution.begin_pre()?;
    let mut index = 0;
    let mut retained_args: Option<Predictions> = None;
    while index < execution.pre_count() {
        retained_args = Some(out.clone());
        out = execution.pre(index, &conditions, &context, out)?;
        index += 1;
    }
    let cond = prediction(&out, 0)?;
    let uncond = prediction(&out, 1)?;
    let result = cfg(execution, context, &cond, &uncond);
    drop(retained_args);
    result
}
pub struct Dual<'a, C> {
    pub x: &'a Tensor,
    pub timestep: &'a Tensor,
    pub options: &'a Options,
    pub positive: Option<&'a C>,
    pub middle: Option<&'a C>,
    pub negative: Option<&'a C>,
    pub cfg1: f64,
    pub cfg2: f64,
    pub nested: bool,
    pub disable_cfg1: bool,
}
pub fn dual<C>(execution: &mut dyn Evaluator<C>, input: Dual<'_, C>) -> Result<Tensor, Error> {
    let Dual {
        x,
        timestep,
        options,
        positive,
        mut middle,
        mut negative,
        cfg1,
        cfg2,
        nested,
        disable_cfg1,
    } = input;
    if !nested && !disable_cfg1 && is_close_one(cfg2) {
        negative = None;
        if is_close_one(cfg1) {
            middle = None;
        }
    }
    let out = execution.batch(&[negative, middle, positive], x, timestep, options)?;
    if nested {
        let pred = cfg(
            execution,
            Context {
                x,
                timestep,
                scale: cfg1,
                options,
                cond: positive,
                uncond: middle,
            },
            &prediction(&out, 2)?,
            &prediction(&out, 1)?,
        )?;
        // Source scalar is the left operand in this branch.
        Ok(prediction(&out, 0)?.f_add(&pred.f_sub(&prediction(&out, 0)?)?.f_mul_scalar(cfg2)?)?)
    } else {
        let base = cfg(
            execution,
            Context {
                x,
                timestep,
                scale: cfg2,
                options,
                cond: middle,
                uncond: negative,
            },
            &prediction(&out, 1)?,
            &prediction(&out, 0)?,
        )?;
        Ok(base.f_add(
            &prediction(&out, 2)?
                .f_sub(&prediction(&out, 1)?)?
                .f_mul_scalar(cfg1)?,
        )?)
    }
}

pub struct Perp<'a, C> {
    pub x: &'a Tensor,
    pub timestep: &'a Tensor,
    pub options: &'a Options,
    pub positive: Option<&'a C>,
    pub negative: Option<&'a C>,
    pub empty: Option<&'a C>,
    pub cfg: f64,
    pub negative_scale: f64,
}
pub struct PerpPost<'a, C> {
    pub cfg: CfgInput<'a, C>,
    pub empty_cond: Option<&'a C>,
    pub empty_denoised: &'a Tensor,
}
pub trait PerpEvaluator<C>: Evaluator<C> {
    fn post_perp(
        &mut self,
        index: usize,
        args: PerpPost<'_, C>,
        denoised: Tensor,
    ) -> Result<Tensor, Error>;
}
enum PerpArgsOwner {
    Pre { _predictions: Predictions },
    Post { _denoised: Tensor },
}
pub fn perp<C>(execution: &mut dyn PerpEvaluator<C>, input: Perp<'_, C>) -> Result<Tensor, Error> {
    use crate::execution_value::Value;
    let Perp {
        x,
        timestep,
        options,
        positive,
        mut negative,
        mut empty,
        cfg,
        negative_scale,
    } = input;
    let disabled = options
        .get("disable_cfg1_optimization")?
        .map_or(Ok(Value::Bool(false)), |v| v.snapshot())?;
    // Original == False is intentionally distinct from general truth conversion.
    if crate::conditioning::is_equal(&disabled, &Value::Bool(false))? && negative_scale == 0. {
        negative = None;
        if is_close_one(cfg) {
            empty = None;
        }
    }
    let conditions = [positive, negative, empty];
    let context = Context {
        x,
        timestep,
        scale: cfg,
        options,
        cond: positive,
        uncond: negative,
    };
    let mut out = execution.batch(&conditions, x, timestep, options)?;
    execution.begin_pre()?;
    let mut index = 0;
    let mut retained_args = None;
    while index < execution.pre_count() {
        retained_args = Some(PerpArgsOwner::Pre {
            _predictions: out.clone(),
        });
        out = execution.pre(index, &conditions, &context, out)?;
        index += 1;
    }
    if out.borrow().len() != 3 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: if out.borrow().len() > 3 {
                "too many values to unpack (expected 3)".into()
            } else {
                format!(
                    "not enough values to unpack (expected 3, got {})",
                    out.borrow().len()
                )
            },
        });
    }
    let unpacked = [
        prediction(&out, 0)?,
        prediction(&out, 1)?,
        prediction(&out, 2)?,
    ];
    let mut result = super::advanced_guidance::perp_neg(
        &unpacked[0],
        &unpacked[1],
        &unpacked[2],
        negative_scale,
        cfg,
    )?;
    execution.begin_post()?;
    let mut index = 0;
    while index < execution.post_count() {
        // PerpNeg reuses the same Python `args` local for pre and post hooks;
        // constructing the first post dictionary releases the prior pre args.
        retained_args = Some(PerpArgsOwner::Post {
            _denoised: result.shallow_clone(),
        });
        result = execution.post_perp(
            index,
            PerpPost {
                cfg: CfgInput {
                    context: Context {
                        x,
                        timestep,
                        scale: cfg,
                        options,
                        cond: positive,
                        uncond: negative,
                    },
                    cond_denoised: &unpacked[0],
                    uncond_denoised: &unpacked[1],
                    cond_noise: None,
                    uncond_noise: None,
                },
                empty_cond: empty,
                empty_denoised: &unpacked[2],
            },
            result,
        )?;
        index += 1;
    }
    drop(retained_args);
    Ok(result)
}
