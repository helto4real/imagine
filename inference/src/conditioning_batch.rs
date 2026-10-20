//! Shared _calc_cond_batch core. Models/controls/hooks provide their original
//! operations through typed Rust boundaries; grouping and memory policy live here.
use crate::{
    conditioning::{AreaCondition, Condition},
    execution_value::Value as V,
    native::Error,
    patching::{
        hooks::HookGroup,
        options::{self, Options},
    },
    sampling::locals::SourceLocals,
};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Tensor};

pub type Hooks = Rc<RefCell<HookGroup>>;
pub trait Control {
    fn get_control(
        &self,
        input: &Tensor,
        timestep: &Tensor,
        conditioning: &Options,
        chunks: usize,
        transformer: &Options,
    ) -> Result<Options, Error>;
}
pub trait Gligen {
    fn patch(&self, shape: &[i64], device: Device) -> Result<Options, Error>;
}
pub struct Record {
    pub area: AreaCondition,
    pub conditions: Vec<(String, Condition)>,
    pub hooks: Option<Hooks>,
    pub control: Option<Rc<dyn Control>>,
    pub gligen: Option<Rc<dyn Gligen>>,
    pub uuid: V,
    /// Presence of the source "default" key, regardless of its stored value.
    pub default: bool,
}
struct Prepared {
    input: Tensor,
    mult: Tensor,
    conditions: Vec<(String, Condition)>,
    area: Option<Vec<i64>>,
    hooks: Option<Hooks>,
    control: Option<Rc<dyn Control>>,
    patches: Option<Options>,
    uuid: V,
}
impl Clone for Prepared {
    fn clone(&self) -> Self {
        Self {
            input: self.input.shallow_clone(),
            mult: self.mult.shallow_clone(),
            conditions: self.conditions.clone(),
            area: self.area.clone(),
            hooks: self.hooks.clone(),
            control: self.control.clone(),
            patches: self.patches.clone(),
            uuid: self.uuid.clone(),
        }
    }
}
fn empty_options() -> Options {
    Options::dictionary(vec![])
}
pub struct Call<'a> {
    pub input: &'a Tensor,
    pub timestep: &'a Tensor,
    pub conditioning: &'a Options,
    pub indices: &'a [usize],
}
pub trait Model {
    fn prepare_hook_keyframe(
        &mut self,
        time: &Tensor,
        hooks: &Hooks,
        options: &Options,
    ) -> Result<(), Error>;
    fn prepare_state(&mut self, time: &Tensor, options: &Options) -> Result<(), Error>;
    fn free_memory(&mut self, device: Device) -> Result<f64, Error>;
    fn memory_required(
        &mut self,
        shape: &[i64],
        conditions: &[(String, Vec<V>)],
    ) -> Result<f64, Error>;
    fn apply_hooks(&mut self, hooks: Option<&Hooks>) -> Result<Options, Error>;
    /// Includes the optional model_function_wrapper around apply_model.
    fn apply_model(&mut self, call: Call<'_>, options: &Options) -> Result<Tensor, Error>;
    fn warning(&mut self, message: &str);
    fn source_locals(&mut self, _frame: &SourceLocals, _event: &str) -> Result<(), Error> {
        Ok(())
    }
}
fn prepare(record: &Record, input: &Tensor, time: &Tensor) -> Result<Option<Prepared>, Error> {
    let Some(area) = record.area.prepare(input, time)? else {
        return Ok(None);
    };
    let conditions = record
        .conditions
        .iter()
        .map(|(name, c)| {
            Ok((
                name.clone(),
                c.process(input.size()[0], area.area.as_deref())?,
            ))
        })
        .collect::<Result<_, Error>>()?;
    let patches = record
        .gligen
        .as_ref()
        .map(|g| {
            let patch = g.patch(&area.input.size(), area.input.device())?;
            let out = empty_options();
            out.set(
                "middle_patch",
                Options::List(Rc::new(RefCell::new(vec![patch]))),
            )?;
            Ok::<_, Error>(out)
        })
        .transpose()?;
    Ok(Some(Prepared {
        input: area.input,
        mult: area.multiplier,
        conditions,
        area: area.area,
        hooks: record.hooks.clone(),
        control: record.control.clone(),
        patches,
        uuid: record.uuid.clone(),
    }))
}
fn same_rc<T: ?Sized>(a: &Option<Rc<T>>, b: &Option<Rc<T>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}
fn can_concat(a: &Prepared, b: &Prepared, warning: &mut dyn FnMut(&str)) -> Result<bool, Error> {
    if a.input.size() != b.input.size() || !same_rc(&a.control, &b.control) {
        return Ok(false);
    }
    match (&a.patches, &b.patches) {
        (None, None) => {}
        (Some(a), Some(b)) if a.same_object(b) => {}
        _ => return Ok(false),
    }
    if std::ptr::eq(a, b) {
        return Ok(true);
    }
    if a.conditions.len() != b.conditions.len() {
        return Ok(false);
    }
    for (name, c) in &a.conditions {
        let Some((_, other)) = b.conditions.iter().find(|(key, _)| key == name) else {
            return Ok(false);
        };
        if !c.can_concat(other, warning)? {
            return Ok(false);
        }
    }
    Ok(true)
}
fn crop(tensor: &Tensor, area: Option<&[i64]>) -> Result<Tensor, Error> {
    let mut out = tensor.shallow_clone();
    if let Some(area) = area {
        let dims = area.len() / 2;
        for i in 0..dims {
            out = out.f_narrow(i as i64 + 2, area[i + dims], area[i])?;
        }
    }
    Ok(out)
}
type Group = (Option<Hooks>, Vec<(Prepared, usize)>);
fn append(groups: &mut Vec<Group>, row: Prepared, index: usize) {
    if let Some((_, rows)) = groups
        .iter_mut()
        .find(|(hooks, _)| same_rc(hooks, &row.hooks))
    {
        rows.push((row, index));
    } else {
        groups.push((row.hooks.clone(), vec![(row, index)]));
    }
}
fn finalize_defaults(
    model: &mut dyn Model,
    groups: &mut Vec<Group>,
    defaults: &[Vec<&Record>],
    input: &Tensor,
    time: &Tensor,
    options: &Options,
) -> Result<(), Error> {
    let mut mults = defaults
        .iter()
        .map(|_| input.f_ones_like())
        .collect::<Result<Vec<_>, _>>()?;
    for (_, rows) in groups.iter() {
        for (row, index) in rows {
            if defaults[*index].is_empty() {
                continue;
            }
            let mut mult = crop(&mults[*index], row.area.as_deref())?;
            let _ = mult.f_sub_(&row.mult)?;
        }
    }
    for (index, mult) in mults.iter_mut().enumerate() {
        if defaults[index].is_empty() {
            continue;
        }
        let _ = mult.f_relu_()?;
        if mult.f_max()?.double_value(&[]) == 0. {
            continue;
        }
        for record in &defaults[index] {
            let Some(mut row) = prepare(record, input, time)? else {
                continue;
            };
            row.mult = mult.shallow_clone();
            if let Some(hooks) = &row.hooks {
                model.prepare_hook_keyframe(time, hooks, options)?;
            }
            append(groups, row, index);
        }
    }
    Ok(())
}
pub fn calculate(
    model: &mut dyn Model,
    conditions: &[Option<&[Record]>],
    input: &Tensor,
    time: &Tensor,
    options: &Options,
) -> Result<Vec<Tensor>, Error> {
    let mut outputs = Vec::new();
    let mut counts = Vec::new();
    let mut groups = Vec::new();
    let mut defaults = Vec::new();
    let mut has_default = false;
    for (index, records) in conditions.iter().enumerate() {
        outputs.push(input.f_zeros_like()?);
        counts.push(input.f_ones_like()?.f_mul_scalar(1e-37)?);
        let mut default = Vec::new();
        if let Some(records) = records {
            for record in *records {
                if record.default {
                    default.push(record);
                    has_default = true;
                    continue;
                }
                let Some(row) = prepare(record, input, time)? else {
                    continue;
                };
                if let Some(hooks) = &row.hooks {
                    model.prepare_hook_keyframe(time, hooks, options)?;
                }
                append(&mut groups, row, index);
            }
        }
        defaults.push(default);
    }
    if has_default {
        finalize_defaults(model, &mut groups, &defaults, input, time, options)?;
    }
    model.prepare_state(time, options)?;
    // Source while-loop variables outlive each iteration, including the old
    // model output, until their respective assignments during the next batch.
    let mut frame = SourceLocals::default();
    let mut previous_c: Option<Options> = None;
    let mut previous_p: Option<Prepared> = None;
    let mut previous_first: Option<Prepared> = None;
    let mut previous_transformer: Option<Options> = None;
    for (hooks, mut rows) in groups {
        while !rows.is_empty() {
            let first = &rows[0].0;
            previous_first = Some(first.clone());
            let shape = first.input.size();
            let mut compatible = Vec::new();
            for (index, (row, _)) in rows.iter().enumerate() {
                if can_concat(row, first, &mut |message| model.warning(message))? {
                    compatible.push(index);
                }
            }
            compatible.reverse();
            let mut batch = compatible[..compatible.len().min(1)].to_vec();
            model.source_locals(&frame, "free_memory")?;
            let free = model.free_memory(input.device())?;
            for divisor in 1..=compatible.len() {
                let candidates = &compatible[..compatible.len() / divisor];
                let mut shape = shape.clone();
                shape[0] *= candidates.len() as i64;
                let mut shapes: Vec<(String, Vec<V>)> = Vec::new();
                for index in candidates {
                    for (name, c) in &rows[*index].0.conditions {
                        let size = c.size()?;
                        if let Some((_, values)) = shapes.iter_mut().find(|(key, _)| key == name) {
                            values.push(size);
                        } else {
                            shapes.push((name.clone(), vec![size]));
                        }
                    }
                }
                if model.memory_required(&shape, &shapes)? * 1.5 < free {
                    batch = candidates.to_vec();
                    break;
                }
            }
            drop(previous_c.take());
            drop(previous_p.take());
            frame.keep_list("mult", &[]);
            frame.remove("input_x");
            let mut selected = Vec::new();
            for index in batch {
                selected.push(rows.remove(index));
            }
            let indices = selected.iter().map(|(_, index)| *index).collect::<Vec<_>>();
            let inputs = selected
                .iter()
                .map(|(row, _)| row.input.shallow_clone())
                .collect::<Vec<_>>();
            frame.keep_list(
                "mult",
                &selected
                    .iter()
                    .map(|(row, _)| row.mult.shallow_clone())
                    .collect::<Vec<_>>(),
            );
            let batched = frame.keep("input_x", Tensor::f_cat(&inputs, 0)?);
            drop(inputs);
            let conditioning = empty_options();
            if let Some((first, _)) = selected.first() {
                for (name, c) in &first.conditions {
                    let others = selected
                        .iter()
                        .skip(1)
                        .map(|(row, _)| {
                            &row.conditions
                                .iter()
                                .find(|(key, _)| key == name)
                                .unwrap()
                                .1
                        })
                        .collect::<Vec<_>>();
                    conditioning.set(name, c.concat_options(&others)?)?;
                }
            }
            let uuids = selected.iter().map(|(row, _)| row.uuid.clone()).collect();
            let areas = selected
                .iter()
                .map(|(row, _)| row.area.clone())
                .collect::<Vec<_>>();
            let mults = selected
                .iter()
                .map(|(row, _)| row.mult.shallow_clone())
                .collect::<Vec<_>>();
            previous_p = selected.pop().map(|(row, _)| row);
            drop(selected);
            let last = previous_p
                .as_ref()
                .ok_or_else(|| Error::Native("Empty compatible conditioning batch".into()))?;
            previous_c = Some(conditioning.clone());
            let timestep = frame.keep("timestep_", Tensor::f_cat(&vec![time; indices.len()], 0)?);
            let transformer = model.apply_hooks(hooks.as_ref())?;
            previous_transformer = Some(transformer.clone());
            if let Some(extra) = options.get("transformer_options")? {
                options::merge_nested_dicts(&transformer, &extra, false)?;
            }
            if let Some(patches) = &last.patches {
                let base = transformer.get("patches")?.unwrap_or_else(empty_options);
                transformer.set(
                    "patches",
                    options::merge_nested_dicts(&base, patches, true)?,
                )?;
            }
            transformer.set(
                "cond_or_uncond",
                Options::from_value(V::List(indices.iter().map(|n| V::Int(*n as i64)).collect())),
            )?;
            transformer.set("uuids", Options::from_value(V::List(uuids)))?;
            transformer.set(
                "sigmas",
                Options::from_value(V::tensor(time.shallow_clone())),
            )?;
            conditioning.set("transformer_options", transformer.clone())?;
            if let Some(control) = &last.control {
                conditioning.set(
                    "control",
                    control.get_control(
                        &batched,
                        &timestep,
                        &conditioning,
                        indices.len(),
                        &transformer,
                    )?,
                )?;
            }
            model.source_locals(&frame, "apply_model")?;
            let output = model.apply_model(
                Call {
                    input: &batched,
                    timestep: &timestep,
                    conditioning: &conditioning,
                    indices: &indices,
                },
                options,
            )?;
            let chunks = output.f_chunk(indices.len() as i64, 0)?;
            frame.keep_list("output", &chunks);
            drop(output);
            for (index, (target, chunk)) in indices.iter().zip(&chunks).enumerate() {
                let mut out = crop(&outputs[*target], areas[index].as_deref())?;
                let mut count = crop(&counts[*target], areas[index].as_deref())?;
                let _ = out.f_add_(&chunk.f_mul(&mults[index])?)?;
                let _ = count.f_add_(&mults[index])?;
                if areas[index].is_some() {
                    let _ = frame.keep("out_c", out);
                    let _ = frame.keep("out_cts", count);
                }
            }
        }
    }
    // Explicit retained owners, independent of whether an observer is installed.
    drop(previous_c);
    drop(previous_p);
    drop(previous_first);
    drop(previous_transformer);
    drop(frame);
    for (output, count) in outputs.iter_mut().zip(counts) {
        let _ = output.f_div_(&count)?;
    }
    Ok(outputs)
}
