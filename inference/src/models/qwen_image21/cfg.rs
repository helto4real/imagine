//! Frozen full-area Qwen CFG conditioning, memory-fit batching and aggregation.
//! Area/mask/control/hook wrappers remain separate shared sampler contracts.
use super::{cfg_combine, cfg1_shortcut};
use crate::native::Error;
use tch::{Device, Tensor};

pub struct FullAreaCondition {
    pub context: Tensor,
    pub attention_mask: Option<Tensor>,
    pub reference_latents: Option<Vec<Tensor>>,
    pub image_slots: Option<Vec<i64>>,
    /// `c_concat` (CONDNoiseShape): extra input channels concatenated to the latent.
    pub concat: Option<Tensor>,
}
fn repeat_batch(tensor: &Tensor, batch: i64) -> Result<Tensor, Error> {
    let size = tensor.size()[0];
    if size > batch {
        Ok(tensor.f_narrow(0, 0, batch)?)
    } else if size < batch {
        if size == 0 {
            return Err(Error::ReferenceException {
                class: "ZeroDivisionError",
                message: "division by zero".into(),
            });
        }
        let mut repeats = vec![1; tensor.dim()];
        repeats[0] = (batch as f64 / size as f64).ceil() as i64;
        Ok(tensor.f_repeat(&repeats)?.f_narrow(0, 0, batch)?)
    } else {
        Ok(tensor.shallow_clone())
    }
}
impl FullAreaCondition {
    fn process(&self, batch: i64) -> Result<Self, Error> {
        Ok(Self {
            context: repeat_batch(&self.context, batch)?,
            attention_mask: self
                .attention_mask
                .as_ref()
                .map(|m| repeat_batch(m, batch))
                .transpose()?,
            reference_latents: self
                .reference_latents
                .as_ref()
                .map(|refs| refs.iter().map(|r| repeat_batch(r, batch)).collect())
                .transpose()?,
            image_slots: self.image_slots.clone(),
            concat: self
                .concat
                .as_ref()
                .map(|c| repeat_batch(c, batch))
                .transpose()?,
        })
    }
    fn can_concat(&self, other: &Self) -> bool {
        fn regular(a: &Tensor, b: &Tensor) -> bool {
            a.size() == b.size() && a.device() == b.device()
        }
        regular(&self.context, &other.context)
            && match (&self.attention_mask, &other.attention_mask) {
                (None, None) => true,
                (Some(a), Some(b)) => regular(a, b),
                _ => false,
            }
            && match (&self.reference_latents, &other.reference_latents) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.size() == b.size())
                }
                _ => false,
            }
            && self.image_slots == other.image_slots
            && match (&self.concat, &other.concat) {
                (None, None) => true,
                (Some(a), Some(b)) => regular(a, b),
                _ => false,
            }
    }
    fn concat(conditions: &[&Self]) -> Result<Self, Error> {
        let first = conditions[0];
        let cat = |values: Vec<&Tensor>| Tensor::f_cat(&values, 0).map_err(Error::from);
        Ok(Self {
            context: cat(conditions.iter().map(|c| &c.context).collect())?,
            attention_mask: first
                .attention_mask
                .as_ref()
                .map(|_| {
                    cat(conditions
                        .iter()
                        .map(|c| c.attention_mask.as_ref().unwrap())
                        .collect())
                })
                .transpose()?,
            reference_latents: first
                .reference_latents
                .as_ref()
                .map(|refs| {
                    (0..refs.len())
                        .map(|i| {
                            cat(conditions
                                .iter()
                                .map(|c| &c.reference_latents.as_ref().unwrap()[i])
                                .collect())
                        })
                        .collect()
                })
                .transpose()?,
            image_slots: first.image_slots.clone(),
            concat: first
                .concat
                .as_ref()
                .map(|_| {
                    cat(conditions
                        .iter()
                        .map(|c| c.concat.as_ref().unwrap())
                        .collect())
                })
                .transpose()?,
        })
    }
}
pub trait FullAreaExecution {
    fn free_memory(&mut self, device: Device) -> Result<f64, Error>;
    fn memory_required(&self, input: &[i64], conditions: &[&FullAreaCondition]) -> f64;
    /// Returns denoised float predictions after BaseModel's parameterization.
    fn apply_model(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        condition: &FullAreaCondition,
        cond_or_uncond: &[usize],
    ) -> Result<Tensor, Error>;
}
/// Default two-branch full-area `sampling_function` including CFG=1 shortcut.
/// Qwen uses CONDRegular for context: unequal text lengths cannot be batched.
pub fn full_area_cfg<E: FullAreaExecution>(
    x: &Tensor,
    timestep: &Tensor,
    positive: &FullAreaCondition,
    negative: Option<&FullAreaCondition>,
    scale: f64,
    disable_cfg1_optimization: bool,
    execution: &mut E,
) -> Result<Tensor, Error> {
    let negative = if cfg1_shortcut(scale, disable_cfg1_optimization) {
        None
    } else {
        negative
    };
    let conditions = [Some(positive), negative]
        .into_iter()
        .map(|c| c.map(|c| c.process(x.size()[0])).transpose())
        .collect::<Result<Vec<_>, Error>>()?;
    let mut outputs = [x.f_zeros_like()?, x.f_zeros_like()?];
    let mut counts = [
        x.f_ones_like()?.f_mul_scalar(1e-37)?,
        x.f_ones_like()?.f_mul_scalar(1e-37)?,
    ];
    let mult = x.f_ones_like()?.f_mul_scalar(1.)?;
    let mut pending = conditions
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().map(|_| i))
        .collect::<Vec<_>>();
    while !pending.is_empty() {
        let first = conditions[pending[0]].as_ref().unwrap();
        let candidates = pending
            .iter()
            .copied()
            .filter(|&i| first.can_concat(conditions[i].as_ref().unwrap()))
            .rev()
            .collect::<Vec<_>>();
        let mut selected = &candidates[..1];
        let free = execution.free_memory(x.device())?;
        for divisor in 1..=candidates.len() {
            let group = &candidates[..candidates.len() / divisor];
            let mut shape = x.size();
            shape[0] *= group.len() as i64;
            let c = group
                .iter()
                .map(|&i| conditions[i].as_ref().unwrap())
                .collect::<Vec<_>>();
            if execution.memory_required(&shape, &c) * 1.5 < free {
                selected = group;
                break;
            }
        }
        let input = Tensor::f_cat(&vec![x; selected.len()], 0)?;
        let condition = FullAreaCondition::concat(
            &selected
                .iter()
                .map(|&i| conditions[i].as_ref().unwrap())
                .collect::<Vec<_>>(),
        )?;
        let time = Tensor::f_cat(&vec![timestep; selected.len()], 0)?;
        let output = execution
            .apply_model(&input, &time, &condition, selected)?
            .f_chunk(selected.len() as i64, 0)?;
        for (&branch, output) in selected.iter().zip(output) {
            let _ = outputs[branch].f_add_(&output.f_mul(&mult)?)?;
            let _ = counts[branch].f_add_(&mult)?;
        }
        pending.retain(|i| !selected.contains(i));
    }
    let _ = outputs[0].f_div_(&counts[0])?;
    let _ = outputs[1].f_div_(&counts[1])?;
    cfg_combine(&outputs[0], &outputs[1], scale)
}
