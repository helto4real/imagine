//! Ordered conditioning preprocessing from comfy.samplers. Records retain
//! source dict/list aliases; replacements copy only the source containers.
use crate::{
    conditioning::add_area_dims,
    conditioning_batch::Hooks,
    execution_value::Value as V,
    native::Error,
    patching::{hooks::HookGroup, options::Options},
};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Kind, Tensor};

fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn required(record: &Options, name: &str) -> Result<Options, Error> {
    record
        .get(name)?
        .ok_or_else(|| error("KeyError", format!("'{name}'")))
}
fn atom(v: V) -> Options {
    Options::Atom(v)
}
fn is_none(v: &Options) -> bool {
    matches!(v, Options::Atom(V::None))
}
fn number(v: &Options) -> Result<f64, Error> {
    match v.snapshot()? {
        V::Float(x) => Ok(x),
        V::Int(x) => Ok(x as f64),
        V::UInt(x) => Ok(x as f64),
        V::Bool(x) => Ok(u8::from(x) as f64),
        V::Tensor(_) => Ok(v.snapshot()?.as_tensor()?.f_double_value(&[])?),
        _ => Err(error(
            "TypeError",
            "conditioning numeric field requires a number",
        )),
    }
}
fn truth(v: &Options) -> Result<bool, Error> {
    if let Some(value) = v.metadata_truth() {
        return value;
    }
    Ok(match v {
        Options::Atom(V::None) => false,
        Options::Atom(V::Bool(x)) => *x,
        Options::Atom(V::Int(x)) => *x != 0,
        Options::Atom(V::UInt(x)) => *x != 0,
        Options::Atom(V::Float(x)) => *x != 0.,
        Options::Atom(V::Text(x)) => !x.is_empty(),
        Options::Atom(V::Tensor(_)) => v.snapshot()?.as_tensor()?.f_is_nonzero()?,
        Options::Dict(v) => !v.borrow().is_empty(),
        Options::List(v) => !v.borrow().is_empty(),
        Options::Tuple(v) => !v.is_empty(),
        _ => true,
    })
}
fn sequence(v: &Options) -> Result<Vec<V>, Error> {
    match v.snapshot()? {
        V::List(values) | V::Tuple(values) => Ok(values),
        _ => Err(error("TypeError", "conditioning area requires a sequence")),
    }
}
fn area(v: &Options) -> Result<Vec<i64>, Error> {
    sequence(v)?
        .into_iter()
        .map(|x| match x {
            V::Int(x) => Ok(x),
            V::Bool(x) => Ok(i64::from(x)),
            _ => Err(error(
                "TypeError",
                "conditioning absolute area requires integers",
            )),
        })
        .collect()
}
fn tuple_area(v: Vec<i64>) -> Options {
    Options::from_value(V::Tuple(v.into_iter().map(V::Int).collect()))
}
fn round(x: f64) -> Result<i64, Error> {
    if x.is_nan() {
        return Err(error("ValueError", "cannot convert float NaN to integer"));
    }
    if x.is_infinite() {
        return Err(error(
            "OverflowError",
            "cannot convert float infinity to integer",
        ));
    }
    let x = x.round_ties_even();
    if x < i64::MIN as f64 || x >= -(i64::MIN as f64) {
        return Err(Error::BindingGap {
            backend: "conditioning".into(),
            operation: "arbitrary precision percentage area".into(),
        });
    }
    Ok(x as i64)
}

/// The source returns a single empty tensor for numel()==0, otherwise a tuple.
pub fn get_mask_aabb(masks: &Tensor) -> Result<V, Error> {
    let device = masks.device();
    if masks.numel() == 0 {
        return Ok(V::tensor(Tensor::f_zeros([0, 4], (Kind::Int, device))?));
    }
    let b = *masks
        .size()
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let boxes = Tensor::f_zeros([b, 4], (Kind::Int, device))?;
    let empty = Tensor::f_zeros([b], (Kind::Bool, device))?;
    for i in 0..b {
        let mask = masks.f_select(0, i)?;
        if mask.numel() == 0 {
            continue;
        }
        if !mask.f_ne(0)?.f_max()?.f_is_nonzero()? {
            let _ = empty.f_select(0, i)?.f_fill_(1)?;
            continue;
        }
        let coords = mask.f_nonzero()?;
        let count = mask.dim();
        if count != 2 {
            return Err(error(
                "ValueError",
                if count > 2 {
                    "too many values to unpack (expected 2)".into()
                } else {
                    format!("not enough values to unpack (expected 2, got {count})")
                },
            ));
        }
        let y = coords.f_select(1, 0)?;
        let x = coords.f_select(1, 1)?;
        for (j, value) in [x.f_min()?, y.f_min()?, x.f_max()?, y.f_max()?]
            .iter()
            .enumerate()
        {
            boxes
                .f_select(0, i)?
                .f_select(0, j as i64)?
                .f_copy_(value)?;
        }
    }
    Ok(V::Tuple(vec![V::tensor(boxes), V::tensor(empty)]))
}

/// common_upscale's bilinear/none branch used by mask preprocessing, including
/// its video reshape/movedim and restoration order.
fn upscale_mask(mask: &Tensor, width: i64, height: i64) -> Result<Tensor, Error> {
    let original = mask.size();
    let mut samples = mask.shallow_clone();
    if original.len() > 4 {
        samples = samples
            .f_reshape([
                original[0],
                original[1],
                -1,
                original[original.len() - 2],
                original[original.len() - 1],
            ])?
            .f_movedim_int(2, 1)?
            .f_reshape([
                -1,
                original[1],
                original[original.len() - 2],
                original[original.len() - 1],
            ])?;
    }
    if samples.dim() != 4 {
        let spatial = &samples.size()[2..];
        return Err(error(
            "ValueError",
            format!(
                "Input and output must have the same number of spatial dimensions, but got input with spatial dimensions of {spatial:?} and output size of ({height}, {width}). Please provide input tensor in (N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format."
            ),
        ));
    }
    let out = samples.f_upsample_bilinear2d([height, width], false, None, None)?;
    if original.len() == 4 {
        return Ok(out);
    }
    let mut shape = original[..original.len() - 2].to_vec();
    shape.extend([height, width]);
    Ok(out
        .f_reshape([original[0], -1, original[1], height, width])?
        .f_movedim_int(2, 1)?
        .f_reshape(&shape)?)
}
/// torch.Size compares equal to a tuple of dimensions, but not to a list.
/// The deprecated source wrapper deliberately passes a list and resamples even
/// matching masks; process_conds passes noise.shape[2:] instead.
#[derive(Clone, Copy)]
pub enum Dimensions<'a> {
    Size(&'a [i64]),
    Tuple(&'a [i64]),
    List(&'a [i64]),
}
impl<'a> Dimensions<'a> {
    pub(crate) fn values(self) -> &'a [i64] {
        match self {
            Self::Size(v) | Self::Tuple(v) | Self::List(v) => v,
        }
    }
}
pub fn resolve_areas_and_cond_masks_multidim(
    records: &mut [Options],
    dimensions: Dimensions<'_>,
    device: Device,
) -> Result<(), Error> {
    let dims = dimensions.values();
    for slot in records {
        let mut record = slot.clone();
        if let Some(value) = record.get("area")? {
            let values = sequence(&value)?;
            let first = values.first().ok_or_else(|| {
                error(
                    "IndexError",
                    if matches!(value, Options::Tuple(_)) {
                        "tuple index out of range"
                    } else {
                        "list index out of range"
                    },
                )
            })?;
            if matches!(first,V::Text(s) if s=="percentage") {
                let a = &values[1..];
                let n = a.len() / 2;
                let mut absolute = Vec::new();
                for (d, size) in dims.iter().enumerate() {
                    let v = a
                        .get(d)
                        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
                    absolute.push(1.max(round(number(&atom(v.clone()))? * *size as f64)?));
                }
                for (d, size) in dims.iter().enumerate() {
                    let v = a
                        .get(d + n)
                        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
                    absolute.push(round(number(&atom(v.clone()))? * *size as f64)?);
                }
                record = record.shallow_copy_dict()?;
                record.set("area", tuple_area(absolute))?;
                *slot = record.clone();
            }
        }
        if let Some(value) = record.get("mask")? {
            let input = value.snapshot()?.as_tensor()?;
            let mut same_object = input.device() == device;
            let mut mask = input.f_to(device)?;
            let modified = record.shallow_copy_dict()?;
            if mask.dim() == dims.len() {
                mask = mask.f_unsqueeze(0)?;
                same_object = false;
            }
            if matches!(dimensions, Dimensions::List(_))
                || mask.size().get(1..).unwrap_or(&[]) != dims
            {
                same_object = false;
                let width = *dims
                    .last()
                    .ok_or_else(|| error("IndexError", "list index out of range"))?;
                let height = *dims
                    .get(dims.len().wrapping_sub(2))
                    .ok_or_else(|| error("IndexError", "list index out of range"))?;
                mask = if mask.dim() < 4 {
                    upscale_mask(&mask.f_unsqueeze(1)?, width, height)?.f_squeeze_dim(1)?
                } else {
                    upscale_mask(&mask, width, height)?
                };
            }
            if modified
                .get("set_area_to_bounds")?
                .as_ref()
                .map(truth)
                .transpose()?
                .unwrap_or(false)
            {
                let bounds = mask.f_abs()?.f_max_dim(0, false)?.0.f_unsqueeze(0)?;
                let V::Tuple(values) = get_mask_aabb(&bounds)? else {
                    return Err(error(
                        "ValueError",
                        "not enough values to unpack (expected 2, got 0)",
                    ));
                };
                let absolute = if values[1].as_tensor()?.f_select(0, 0)?.f_is_nonzero()? {
                    vec![8, 8, 0, 0]
                } else {
                    let b = values[0].as_tensor()?.f_select(0, 0)?;
                    // Keep the source Int32 scalar subtraction/addition before
                    // Python max/int conversion (including wrapping behavior).
                    let h = b
                        .f_select(0, 3)?
                        .f_sub(&b.f_select(0, 1)?)?
                        .f_add_scalar(1)?
                        .int64_value(&[]);
                    let w = b
                        .f_select(0, 2)?
                        .f_sub(&b.f_select(0, 0)?)?
                        .f_add_scalar(1)?
                        .int64_value(&[]);
                    vec![h.max(8), w.max(8), b.int64_value(&[1]), b.int64_value(&[0])]
                };
                modified.set("area", tuple_area(absolute))?;
            }
            modified.set(
                "mask",
                if same_object {
                    value
                } else {
                    atom(V::tensor(mask))
                },
            )?;
            *slot = modified;
        }
    }
    Ok(())
}
pub fn resolve_areas_and_cond_masks(
    records: &mut [Options],
    h: i64,
    w: i64,
    device: Device,
    warning: &mut dyn FnMut(&str),
) -> Result<(), Error> {
    warning(
        "WARNING: The comfy.samplers.resolve_areas_and_cond_masks function is deprecated please use the resolve_areas_and_cond_masks_multidim one instead.",
    );
    resolve_areas_and_cond_masks_multidim(records, Dimensions::List(&[h, w]), device)
}
fn inside(original_a: &Options, original_b: &Options) -> Result<bool, Error> {
    let a = area(original_a)?;
    let b = area(original_b)?;
    if (a.len() / 2 < b.len() / 2 && matches!(original_a, Options::Tuple(_)))
        || (b.len() / 2 < a.len() / 2 && matches!(original_b, Options::Tuple(_)))
    {
        return Err(error(
            "TypeError",
            "can only concatenate list (not \"tuple\") to list",
        ));
    }
    let a = add_area_dims(&a, b.len() / 2);
    let b = add_area_dims(&b, a.len() / 2);
    let (na, nb) = (a.len() / 2, b.len() / 2);
    Ok((0..na.min(nb)).all(|i| {
        a[na + i] >= b[nb + i]
            && (a[i] as i128 + a[na + i] as i128) <= (b[i] as i128 + b[nb + i] as i128)
    }))
}
pub fn create_cond_with_same_area_if_none(
    records: &mut Vec<Options>,
    source: &Options,
) -> Result<(), Error> {
    let Some(source_area) = source.get("area")? else {
        return Ok(());
    };
    let mut smallest: Option<&Options> = None;
    for record in records.iter() {
        if let Some(candidate) = record.get("area")? {
            let b = area(&candidate)?;
            if inside(&source_area, &candidate)? {
                let replace = match smallest {
                    None => true,
                    Some(old) => match old.get("area")? {
                        None => true,
                        Some(old) => {
                            let old = area(&old)?;
                            old[..old.len() / 2]
                                .iter()
                                .map(|n| *n as i128)
                                .product::<i128>()
                                > b[..b.len() / 2]
                                    .iter()
                                    .map(|n| *n as i128)
                                    .product::<i128>()
                        }
                    },
                };
                if replace {
                    smallest = Some(record);
                }
            }
        } else if smallest.is_none() {
            smallest = Some(record);
        }
    }
    let Some(smallest) = smallest else {
        return Ok(());
    };
    if let Some(b) = smallest.get("area")?
        && crate::conditioning::is_equal(&b.snapshot()?, &source_area.snapshot()?)?
    {
        return Ok(());
    }
    let output = source.shallow_copy_dict()?;
    output.set(
        "model_conds",
        required(smallest, "model_conds")?.shallow_copy_dict()?,
    )?;
    records.push(output);
    Ok(())
}
pub fn calculate_start_end_timesteps(
    records: &mut [Options],
    percent_to_sigma: &mut dyn FnMut(&Options) -> Result<Options, Error>,
) -> Result<(), Error> {
    for slot in records {
        let mut start = None;
        let mut end = None;
        if let Some(clip) = slot.get("clip_start_percent")? {
            let other = slot.get("start_percent")?.unwrap_or(atom(V::Float(0.)));
            let v = if number(&other)? > number(&clip)? {
                other
            } else {
                clip
            };
            start = Some(percent_to_sigma(&v)?);
            let clip = required(slot, "clip_end_percent")?;
            let other = slot.get("end_percent")?.unwrap_or(atom(V::Float(1.)));
            let v = if number(&other)? < number(&clip)? {
                other
            } else {
                clip
            };
            end = Some(percent_to_sigma(&v)?);
        } else {
            if let Some(v) = slot.get("start_percent")? {
                start = Some(percent_to_sigma(&v)?);
            }
            if let Some(v) = slot.get("end_percent")? {
                end = Some(percent_to_sigma(&v)?);
            }
        }
        let start = start.filter(|v| !is_none(v));
        let end = end.filter(|v| !is_none(v));
        if start.is_some() || end.is_some() {
            let out = slot.shallow_copy_dict()?;
            if let Some(start) = start {
                out.set("timestep_start", start)?;
            }
            if let Some(end) = end {
                out.set("timestep_end", end)?;
            }
            *slot = out;
        }
    }
    Ok(())
}
pub type Fill<'a> = dyn FnMut(&[Options], usize) -> Result<Options, Error> + 'a;
pub fn apply_empty_x_to_equal_area(
    positive: &[Options],
    negative: &mut Vec<Options>,
    name: &str,
    fill: &mut Fill<'_>,
) -> Result<(), Error> {
    let mut values = Vec::new();
    let mut other = Vec::new();
    let mut has_value = false;
    for record in positive {
        if record.get("area")?.is_none()
            && let Some(v) = record.get(name)?.filter(|v| !is_none(v))
        {
            values.push(v);
        }
    }
    for (i, record) in negative.iter().enumerate() {
        if record.get("area")?.is_none() {
            if record.get(name)?.is_some_and(|v| !is_none(&v)) {
                has_value = true;
            } else {
                other.push((record.clone(), i));
            }
        }
    }
    if has_value {
        return Ok(());
    }
    for i in 0..values.len() {
        if other.is_empty() {
            return Err(error("ZeroDivisionError", "integer modulo by zero"));
        }
        let (original, index) = &other[i % other.len()];
        let append = original.get(name)?.is_some_and(|v| !is_none(&v));
        let output = original.shallow_copy_dict()?;
        output.set(name, fill(&values, i)?)?;
        if append {
            negative.push(output)
        } else {
            negative[*index] = output;
        }
    }
    Ok(())
}
pub fn encode_model_conds(
    records: &mut [Options],
    noise: &Tensor,
    device: Device,
    prompt_type: &str,
    kwargs: &Options,
    model: &mut dyn FnMut(&Options) -> Result<Options, Error>,
) -> Result<(), Error> {
    for slot in records {
        let params = slot.shallow_copy_dict()?;
        params.set("device", atom(V::Device(device)))?;
        params.set("noise", atom(V::tensor(noise.shallow_clone())))?;
        let shape = noise.size();
        let width = if shape.len() >= 4 {
            atom(V::Int(shape[3] * 8))
        } else {
            atom(V::None)
        };
        let width = params.get("width")?.unwrap_or(width);
        params.set("width", width)?;
        // Python evaluates the height default even when a value already exists.
        let height = shape
            .get(2)
            .ok_or_else(|| error("IndexError", "tuple index out of range"))?
            * 8;
        let height = params.get("height")?.unwrap_or(atom(V::Int(height)));
        params.set("height", height)?;
        let prompt = params
            .get("prompt_type")?
            .unwrap_or(atom(V::Text(prompt_type.into())));
        params.set("prompt_type", prompt)?;
        for (k, v) in kwargs.dict_items()? {
            if params.get(&k)?.is_none() {
                params.set(&k, v)?;
            }
        }
        let extra = model(&params)?;
        let out = slot.shallow_copy_dict()?;
        let conditions = required(&out, "model_conds")?.shallow_copy_dict()?;
        for (k, v) in extra.dict_items()? {
            conditions.set(&k, v)?;
        }
        out.set("model_conds", conditions)?;
        *slot = out;
    }
    Ok(())
}
pub fn hook_option(hooks: Option<Hooks>) -> Options {
    hooks.map_or(atom(V::None), |v| Options::Object(v))
}
fn hook_value(record: &Options) -> Result<Option<Hooks>, Error> {
    let Some(v) = record.get("hooks")? else {
        return Ok(None);
    };
    if is_none(&v) {
        return Ok(None);
    }
    match v {
        Options::Object(v) => v.downcast::<RefCell<HookGroup>>().map(Some).map_err(|_| {
            error(
                "AttributeError",
                "'object' object has no attribute 'is_subset_of'",
            )
        }),
        _ => Err(error(
            "AttributeError",
            "conditioning hooks value has no HookGroup binding",
        )),
    }
}
/// A model/control supplies the source operation through a typed Rust binding.
/// These are required operations, rather than silently optional callbacks.
pub type ControlClones = Vec<(Device, Rc<dyn Control>)>;
pub trait Control {
    fn extra_hooks(&self) -> Result<Vec<Option<Hooks>>, Error>;
    fn pre_run(
        &self,
        model: &Options,
        percent_to_sigma: &mut dyn FnMut(&Options) -> Result<Options, Error>,
    ) -> Result<(), Error>;
    fn multigpu_clones(&self) -> Result<ControlClones, Error>;
}
pub struct ControlBinding(pub Rc<dyn Control>);
pub fn control_option(control: Rc<dyn Control>) -> Options {
    Options::Object(Rc::new(ControlBinding(control)))
}
fn control_value(value: &Options, operation: &str) -> Result<Rc<dyn Control>, Error> {
    if let Options::Object(value) = value
        && let Some(value) = value.downcast_ref::<ControlBinding>()
    {
        return Ok(value.0.clone());
    }
    Err(error(
        "AttributeError",
        format!(
            "'{}' object has no attribute '{operation}'",
            value.typename()
        ),
    ))
}
pub fn preprocess_conds_hooks(groups: &[Vec<Options>]) -> Result<(), Error> {
    type Replacement = (Rc<dyn Control>, Option<Hooks>, Vec<Options>);
    let mut replacements: Vec<Replacement> = Vec::new();
    for record in groups.iter().flatten() {
        if let Some(control) = record.get("control")? {
            let control = control_value(&control, "get_extra_hooks")?;
            if !control.extra_hooks()?.is_empty() {
                let hooks = hook_value(record)?;
                if let Some((_, _, records)) = replacements.iter_mut().find(|(c, h, _)| {
                    Rc::ptr_eq(c, &control)
                        && match (h, &hooks) {
                            (None, None) => true,
                            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
                            _ => false,
                        }
                }) {
                    records.push(record.clone());
                } else {
                    replacements.push((control, hooks, vec![record.clone()]));
                }
            }
        }
    }
    for (control, hooks, records) in replacements {
        let mut groups = control.extra_hooks()?;
        groups.push(hooks);
        if let Some(hooks) = HookGroup::combine_all(&groups, 0)? {
            let value = hook_option(Some(hooks));
            for record in records {
                record.set("hooks", value.clone())?;
            }
        }
    }
    Ok(())
}
pub trait ControlModel {
    fn model_object(&self) -> Options;
    /// get_additional_models_with_key("multigpu"), or an empty list when the
    /// source model has no current_patcher. Duplicate devices are overwritten.
    fn multigpu_models(&mut self) -> Result<Vec<(Device, Options)>, Error>;
    fn percent_to_sigma(&mut self, percent: &Options) -> Result<Options, Error>;
}
pub fn pre_run_control(model: &mut dyn ControlModel, records: &[Options]) -> Result<(), Error> {
    let device_models = model.multigpu_models()?;
    for record in records {
        if let Some(control) = record.get("control")? {
            let control = control_value(&control, "pre_run")?;
            let original = model.model_object();
            control.pre_run(&original, &mut |p| model.percent_to_sigma(p))?;
            for (device, clone) in control.multigpu_clones()? {
                let target = device_models
                    .iter()
                    .rev()
                    .find(|(d, _)| *d == device)
                    .map_or_else(|| original.clone(), |(_, m)| m.clone());
                clone.pre_run(&target, &mut |p| model.percent_to_sigma(p))?;
            }
        }
    }
    Ok(())
}
pub fn filter_registered_hooks_on_conds(
    groups: &[Vec<Options>],
    registered: Option<&Hooks>,
) -> Result<(), Error> {
    let Some(registered) = registered else {
        for record in groups.iter().flatten() {
            record.pop("hooks")?;
        }
        return Ok(());
    };
    let mut replacements: Vec<(Hooks, Vec<Options>)> = Vec::new();
    for record in groups.iter().flatten() {
        if let Some(hooks) = hook_value(record)?
            && !hooks.borrow().is_subset(&registered.borrow())
        {
            if let Some((_, records)) = replacements
                .iter_mut()
                .find(|(old, _)| Rc::ptr_eq(old, &hooks))
            {
                records.push(record.clone());
            } else {
                replacements.push((hooks, vec![record.clone()]));
            }
        }
    }
    for (hooks, records) in replacements {
        let common = hooks.borrow().common(&registered.borrow());
        let value = hook_option(if common.hooks.is_empty() {
            None
        } else {
            Some(Rc::new(RefCell::new(common)))
        });
        for record in records {
            record.set("hooks", value.clone())?;
        }
    }
    Ok(())
}
pub fn get_total_hook_groups_in_conds(groups: &[Vec<Options>]) -> Result<usize, Error> {
    let mut seen: Vec<Option<Hooks>> = Vec::new();
    for record in groups.iter().flatten() {
        let hooks = hook_value(record)?;
        if !seen.iter().any(|old| match (old, &hooks) {
            (None, None) => true,
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }) {
            seen.push(hooks);
        }
    }
    Ok(seen.len())
}

pub trait ConditioningModel: ControlModel {
    fn has_extra_conds(&self) -> bool;
    fn extra_conds(&mut self, params: &Options) -> Result<Options, Error>;
}
#[derive(Clone)]
pub struct Arguments {
    pub latent_image: Options,
    pub denoise_mask: Options,
    pub seed: Options,
    pub latent_shapes: Options,
}
impl Default for Arguments {
    fn default() -> Self {
        Self {
            latent_image: atom(V::None),
            denoise_mask: atom(V::None),
            seed: atom(V::None),
            latent_shapes: atom(V::None),
        }
    }
}
fn list(value: &Options) -> Result<crate::patching::options::List, Error> {
    match value {
        Options::List(v) => Ok(v.clone()),
        _ => Err(error(
            "TypeError",
            format!("'{}' object is not subscriptable", value.typename()),
        )),
    }
}
/// Process a fixed source range while publishing each replacement immediately,
/// so callbacks see earlier records and errors preserve partial mutations.
fn each_record(
    group: &Options,
    operation: &mut dyn FnMut(&mut [Options]) -> Result<(), Error>,
) -> Result<(), Error> {
    let rows = list(group)?;
    let length = rows.borrow().len();
    for i in 0..length {
        let mut one = [rows
            .borrow()
            .get(i)
            .cloned()
            .ok_or_else(|| error("IndexError", "list index out of range"))?];
        let result = operation(&mut one);
        *rows
            .borrow_mut()
            .get_mut(i)
            .ok_or_else(|| error("IndexError", "list assignment index out of range"))? =
            one[0].clone();
        result?;
    }
    Ok(())
}
pub fn process_conds(
    model: &mut dyn ConditioningModel,
    noise: &Tensor,
    conditions: &Options,
    device: Device,
    args: &Arguments,
) -> Result<Options, Error> {
    let keys = conditions
        .dict_items()?
        .into_iter()
        .map(|(k, _)| k)
        .collect::<Vec<_>>();
    for key in &keys {
        let current = required(conditions, key)?;
        let rows = list(&current)?.borrow().clone();
        let group = Options::List(Rc::new(RefCell::new(rows)));
        conditions.set(key, group.clone())?;
        each_record(&group, &mut |rows| {
            resolve_areas_and_cond_masks_multidim(
                rows,
                Dimensions::Size(noise.size().get(2..).unwrap_or(&[])),
                device,
            )
        })?;
    }
    for key in &keys {
        each_record(&required(conditions, key)?, &mut |rows| {
            calculate_start_end_timesteps(rows, &mut |p| model.percent_to_sigma(p))
        })?;
    }
    if model.has_extra_conds() {
        let kwargs = Options::dictionary(vec![
            ("latent_image".into(), args.latent_image.clone()),
            ("denoise_mask".into(), args.denoise_mask.clone()),
            ("seed".into(), args.seed.clone()),
            ("latent_shapes".into(), args.latent_shapes.clone()),
        ]);
        for key in &keys {
            each_record(&required(conditions, key)?, &mut |rows| {
                encode_model_conds(rows, noise, device, key, &kwargs, &mut |params| {
                    model.extra_conds(params)
                })
            })?;
        }
    }
    for key in &keys {
        let rows = list(&required(conditions, key)?)?;
        let mut index = 0;
        while index < rows.borrow().len() {
            let source = rows.borrow()[index].clone();
            index += 1;
            for other in &keys {
                if key != other {
                    let target = list(&required(conditions, other)?)?;
                    let mut values = target.borrow().clone();
                    let result = create_cond_with_same_area_if_none(&mut values, &source);
                    *target.borrow_mut() = values;
                    result?;
                }
            }
        }
    }
    for key in &keys {
        let rows = list(&required(conditions, key)?)?;
        let mut index = 0;
        while index < rows.borrow().len() {
            let record = rows.borrow()[index].clone();
            index += 1;
            if let Some(value) = record.get("hooks")? {
                let hooks = hook_value(&record)?.ok_or_else(|| {
                    error(
                        "AttributeError",
                        format!("'{}' object has no attribute 'hooks'", value.typename()),
                    )
                })?;
                let mut index = 0;
                while index < hooks.borrow().hooks.len() {
                    let hook = hooks.borrow().hooks[index].clone();
                    index += 1;
                    hook.borrow().initialize_timesteps(&mut |p| {
                        model.percent_to_sigma(&atom(V::Float(p)))?.snapshot()
                    })?;
                }
            }
        }
    }
    for key in &keys {
        pre_run_control(model, &list(&required(conditions, key)?)?.borrow().clone())?;
    }
    if let Some(positive) = conditions.get("positive")? {
        let positive = list(&positive)?;
        for key in &keys {
            if key == "positive" {
                continue;
            }
            let all = positive.borrow().clone();
            let mut controls = Vec::new();
            for record in &all {
                let enabled = record
                    .get("control_apply_to_uncond")?
                    .unwrap_or(atom(V::Bool(false)));
                if crate::conditioning::is_equal(&enabled.snapshot()?, &V::Bool(true))? {
                    controls.push(record.clone());
                }
            }
            let target = list(&required(conditions, key)?)?;
            for (source, name) in [(&controls, "control"), (&all, "gligen")] {
                let mut values = target.borrow().clone();
                let result =
                    apply_empty_x_to_equal_area(source, &mut values, name, &mut |items, i| {
                        Ok(items[i].clone())
                    });
                *target.borrow_mut() = values;
                result?;
            }
        }
    }
    Ok(conditions.clone())
}
