//! Shared get_resized_cond body; model/control effects require typed bindings.
use super::{Handler, WindowHandle, WindowList, error, gap};
use crate::{
    conditioning::Condition,
    execution_value::Value as V,
    native::{self, Error},
    patching::{extensions::Original, options::Options},
};
use std::{cell::RefCell, rc::Rc};
use tch::Device;

pub trait ResizeModel {
    fn resize_cond_for_context_window(
        &self,
        key: &str,
        value: &Options,
        window: &WindowHandle,
        input: &V,
        device: Option<Device>,
        retain: WindowList<i64>,
    ) -> Result<Options, Error>;
}
#[derive(Clone, Default)]
pub enum ResizeModelSlot {
    #[default]
    Missing,
    None,
    Bound(Rc<dyn ResizeModel>),
}
pub trait ResizingBindings {
    fn prepare_control_objects(
        &self,
        control: &Options,
        device: Option<Device>,
    ) -> Result<Options, Error>;
    fn info(&self, message: &str) -> Result<(), Error>;
    fn source_locals(&self, _locals: &ResizeLocals, _event: &str) -> Result<(), Error> {
        Ok(())
    }
}
#[derive(Default)]
pub struct ResizeLocals {
    values: Vec<(&'static str, Options)>,
}
impl ResizeLocals {
    fn keep(&mut self, name: &'static str, value: Options) {
        if let Some((_, old)) = self.values.iter_mut().find(|(k, _)| *k == name) {
            *old = value;
        } else {
            self.values.push((name, value));
        }
    }
    fn remove(&mut self, name: &str) {
        if let Some(i) = self.values.iter().position(|(k, _)| *k == name) {
            self.values.remove(i);
        }
    }
    pub fn values(&self) -> impl Iterator<Item = (&str, &Options)> {
        self.values.iter().map(|(k, v)| (*k, v))
    }
}
fn is_none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
fn sequence_len(value: &Options) -> Result<usize, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().len()),
        Options::Tuple(v) => Ok(v.len()),
        _ => Err(gap(
            "context conditioning requires its source sequence binding",
        )),
    }
}
fn sequence_get(value: &Options, index: usize) -> Result<Option<Options>, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().get(index).cloned()),
        Options::Tuple(v) => Ok(v.get(index).cloned()),
        _ => Err(gap(
            "context conditioning requires its source sequence binding",
        )),
    }
}
fn condition(value: &Options) -> Option<&Condition> {
    match value {
        Options::Object(v) => v.downcast_ref::<Condition>(),
        _ => None,
    }
}
fn tensor(value: &Options) -> Result<Option<V>, Error> {
    match value {
        Options::Atom(v @ V::Tensor(_)) => Ok(Some(v.clone())),
        Options::Atom(V::Quantized(_)) => Err(gap(
            "quantized context conditioning requires its source indexing binding",
        )),
        _ => Ok(None),
    }
}
fn cond_tensor(value: &Options) -> Result<Option<V>, Error> {
    match condition(value) {
        Some(c) => tensor(&c.source_value()),
        None => Ok(None),
    }
}
fn to_device(value: &V, device: Option<Device>) -> Result<V, Error> {
    let t = value.as_tensor()?;
    if device.is_none_or(|device| device == t.device()) {
        Ok(value.clone())
    } else {
        Ok(V::tensor(t.f_to_device(device.unwrap())?))
    }
}
fn copied(value: &Options, data: Options) -> Result<Options, Error> {
    let c = condition(value).ok_or_else(|| {
        error(
            "AttributeError",
            format!(
                "'{}' object has no attribute '_copy_with'",
                value.typename()
            ),
        )
    })?;
    Ok(Options::Object(Rc::new(c.copy_with_options(data))))
}
fn default_slicing_matches(value: &V, input: &V, dim: i64) -> Result<bool, Error> {
    let v = value.as_tensor()?;
    let x = input.as_tensor()?;
    Ok(
        (dim < v.dim() as i64 && native::context_size(&v, dim)? == native::context_size(&x, dim)?)
            || ((v.dim() as i64) < dim
                && native::context_size(&v, 0)? == native::context_size(&x, dim)?),
    )
}
impl Handler {
    pub fn get_resized_cond(
        &self,
        bindings: &dyn ResizingBindings,
        input: &Options,
        x: &V,
        window: &WindowHandle,
        device: Option<Device>,
    ) -> Result<Options, Error> {
        if is_none(input) {
            return Ok(input.clone());
        }
        let mut locals = ResizeLocals::default();
        let resized = Options::List(Rc::new(RefCell::new(vec![])));
        locals.keep("resized_cond", resized.clone());
        let mut selected = input.clone();
        if self.settings.borrow().split_conditions && sequence_len(input)? > 1 {
            let region = window.borrow().region_index(sequence_len(input)? as i64)?;
            let (first, last, center) = {
                let w = window.borrow();
                let indices = w.indices.borrow();
                (
                    *indices
                        .first()
                        .ok_or_else(|| error("IndexError", "list index out of range"))?,
                    *indices.last().unwrap(),
                    w.center_ratio,
                )
            };
            bindings.source_locals(&locals, "info")?;
            bindings.info(&format!("Splitting conds to windows; using region {region} for window {first}-{last} with center ratio {center:.3}"))?;
            let value = sequence_get(input, region as usize)?
                .ok_or_else(|| error("IndexError", "list index out of range"))?;
            selected = Options::List(Rc::new(RefCell::new(vec![value])));
        }
        let mut position = 0;
        while let Some(actual) = sequence_get(&selected, position)? {
            position += 1;
            locals.keep("actual_cond", actual.clone());
            let out = actual.shallow_copy_dict()?;
            locals.keep("resized_actual_cond", out.clone());
            let keys = actual.keys_iterator()?;
            while let Some(key) = keys.next_key()? {
                // The source finally deletes this per-key local on success/error.
                let item = actual
                    .get(&key)?
                    .ok_or_else(|| error("KeyError", format!("'{key}'")))?;
                locals.keep("cond_item", item.clone());
                let body: Result<(), Error> = (|| {
                    let replacement = if let Some(value) = tensor(&item)? {
                        let dim = self.settings.borrow().dim;
                        let t = value.as_tensor()?;
                        let sliced = if dim < t.dim() as i64
                            && native::context_size(&t, dim)?
                                == native::context_size(&x.as_tensor()?, dim)?
                        {
                            let sliced = window.borrow().get_tensor(&value, None, None, &[])?;
                            locals.keep("actual_cond_item", Options::Atom(sliced.clone()));
                            sliced
                        } else {
                            value
                        };
                        Options::Atom(to_device(&sliced, device)?)
                    } else if key == "control" {
                        bindings.source_locals(&locals, "control")?;
                        bindings.prepare_control_objects(&item, device)?
                    } else if matches!(item, Options::Dict(_)) {
                        let inner = item.shallow_copy_dict()?;
                        locals.keep("new_cond_item", inner.clone());
                        let items = inner.items_iterator()?;
                        while let Some(pair) = items.next_item()? {
                            let Options::Tuple(pair) = pair else {
                                unreachable!()
                            };
                            let Options::Atom(V::Text(cond_key)) = &pair[0] else {
                                unreachable!()
                            };
                            let cond_key = cond_key.clone();
                            let value = pair[1].clone();
                            drop(pair);
                            locals.keep("cond_value", value.clone());
                            let mut handled = false;
                            // Flatten first, release the registry borrow, then invoke.
                            let callbacks = self.callbacks.borrow().all("resize_cond_item");
                            // Python's list iterator owns the complete flattened
                            // snapshot through loop exit, including past entries.
                            for callback in &callbacks {
                                locals.keep("callback", callback.as_ref().clone());
                                let args = vec![
                                    Options::Atom(V::Text(cond_key.clone())),
                                    value.clone(),
                                    Options::Object(window.clone()),
                                    Options::Atom(x.clone()),
                                    Options::Atom(device.map(V::Device).unwrap_or(V::None)),
                                    inner.clone(),
                                ];
                                let result = if let Options::Object(call) = callback.as_ref()
                                    && let Some(call) = call.downcast_ref::<Original<Options>>()
                                {
                                    bindings.source_locals(&locals, "callback")?;
                                    call(&args, &vec![])?
                                } else {
                                    return Err(error(
                                        "TypeError",
                                        format!("'{}' object is not callable", callback.typename()),
                                    ));
                                };
                                locals.keep("result", result.clone());
                                if !is_none(&result) {
                                    inner.set(&cond_key, result)?;
                                    handled = true;
                                    break;
                                }
                            }
                            drop(callbacks);
                            if !handled {
                                let slot = self.model.borrow().clone();
                                let called = matches!(slot, ResizeModelSlot::Bound(_));
                                let result = match slot {
                                    ResizeModelSlot::Missing => {
                                        return Err(error(
                                            "AttributeError",
                                            "'IndexListContextHandler' object has no attribute '_model'",
                                        ));
                                    }
                                    ResizeModelSlot::None => Options::Atom(V::None),
                                    ResizeModelSlot::Bound(model) => {
                                        let retain =
                                            self.settings.borrow().condition_retain.clone();
                                        bindings.source_locals(&locals, "model")?;
                                        model.resize_cond_for_context_window(
                                            &cond_key, &value, window, x, device, retain,
                                        )?
                                    }
                                };
                                // None _model skips the assignment in the source.
                                if called {
                                    locals.keep("result", result.clone());
                                }
                                if !is_none(&result) {
                                    inner.set(&cond_key, result)?;
                                    handled = true;
                                }
                            }
                            if handled {
                                continue;
                            }
                            let dim = self.settings.borrow().dim;
                            if let Some(t) = tensor(&value)? {
                                if default_slicing_matches(&t, x, dim)? {
                                    inner.set(
                                        &cond_key,
                                        Options::Atom(window.borrow().get_tensor(
                                            &t,
                                            device,
                                            None,
                                            &[],
                                        )?),
                                    )?;
                                }
                            } else if cond_key == "audio_embed"
                                && let Some(t) = cond_tensor(&value)?
                            {
                                locals.keep("audio_cond", Options::Atom(t.clone()));
                                let t_raw = t.as_tensor()?;
                                if t_raw.dim() > 1
                                    && native::context_size(&t_raw, 1)?
                                        == native::context_size(&x.as_tensor()?, dim)?
                                {
                                    inner.set(
                                        &cond_key,
                                        copied(
                                            &value,
                                            Options::Atom(window.borrow().get_tensor(
                                                &t,
                                                device,
                                                Some(1),
                                                &[],
                                            )?),
                                        )?,
                                    )?;
                                }
                            } else if cond_key == "vace_context"
                                && let Some(t) = cond_tensor(&value)?
                            {
                                locals.keep("vace_cond", Options::Atom(t.clone()));
                                let t_raw = t.as_tensor()?;
                                if t_raw.dim() >= 4
                                    && native::context_size(&t_raw, 3)?
                                        == native::context_size(&x.as_tensor()?, dim)?
                                {
                                    let retain = self.settings.borrow().condition_retain.clone();
                                    let sliced = window.borrow().get_tensor(
                                        &t,
                                        device,
                                        Some(3),
                                        &retain.borrow(),
                                    )?;
                                    locals.keep("sliced_vace", Options::Atom(sliced.clone()));
                                    inner.set(&cond_key, copied(&value, Options::Atom(sliced))?)?;
                                }
                            } else if let Some(t) = cond_tensor(&value)? {
                                if default_slicing_matches(&t, x, dim)? {
                                    let retain = self.settings.borrow().condition_retain.clone();
                                    let sliced = window.borrow().get_tensor(
                                        &t,
                                        device,
                                        None,
                                        &retain.borrow(),
                                    )?;
                                    inner.set(&cond_key, copied(&value, Options::Atom(sliced))?)?;
                                }
                            } else if cond_key == "num_video_frames" {
                                let original = condition(&value).ok_or_else(|| {
                                    error(
                                        "AttributeError",
                                        format!(
                                            "'{}' object has no attribute '_copy_with'",
                                            value.typename()
                                        ),
                                    )
                                })?;
                                let new = original.copy_with_options(original.source_value());
                                new.set_value(Options::Atom(V::Int(
                                    window.borrow().context_length as i64,
                                )));
                                inner.set(&cond_key, Options::Object(Rc::new(new)))?;
                            }
                        }
                        inner
                    } else {
                        item.clone()
                    };
                    out.set(&key, replacement)?;
                    Ok(())
                })();
                drop(item);
                locals.remove("cond_item");
                body?;
            }
            let Options::List(values) = &resized else {
                unreachable!()
            };
            values.borrow_mut().push(out);
        }
        bindings.source_locals(&locals, "return")?;
        Ok(resized)
    }
}
