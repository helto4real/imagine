//! CFGGuider.sample entry body. Geometry and model operations are required
//! native bindings; partial failures retain the original finally boundaries.
use super::guider::{GuiderState, SampleArguments};
use crate::{
    execution_value::Value as V,
    native::Error,
    patching::{
        extensions::{Original, WrapperExecutor},
        options::Options,
        options_iterator::SequenceIterator,
    },
};
use std::{any::Any, cell::RefCell, rc::Rc};
pub trait SampleBindings {
    fn empty_sigmas(&self, sigmas: &Options) -> Result<bool, Error>;
    fn is_nested(&self, tensor: &Options) -> Result<bool, Error>;
    fn unbind(&self, tensor: &Options) -> Result<Options, Error>;
    fn shape(&self, tensor: &Options) -> Result<Options, Error>;
    fn shape_tuple(&self, shape: &Options) -> Result<Options, Error>;
    fn pack(&self, latents: &Options) -> Result<[Options; 2], Error>;
    fn length(&self, value: &Options) -> Result<usize, Error>;
    fn index(&self, value: &Options, index: usize) -> Result<Options, Error>;
    fn prefix(&self, value: &Options, length: usize) -> Result<Options, Error>;
    fn append(&self, value: &Options, item: Options) -> Result<(), Error>;
    fn set_index(&self, value: &Options, index: usize, item: Options) -> Result<(), Error>;
    fn ones(&self, shape: &Options) -> Result<Options, Error>;
    fn load_device(&self, patcher: &Options) -> Result<Options, Error>;
    fn prepare_mask(
        &self,
        mask: &Options,
        shape: &Options,
        device: &Options,
    ) -> Result<Options, Error>;
    fn mask_float(&self, mask: &Options) -> Result<Options, Error>;
    fn detail(&self, patcher: &Options, sampler_shapes: &Options) -> Result<(), Error>;
    fn packed_callback(&self, callback: &Options, shapes: &Options) -> Result<Options, Error>;
    fn empty_dictionary(&self) -> Result<Options, Error>;
    fn copy_condition_record(&self, record: &Options) -> Result<Options, Error>;
    fn preprocess_hooks(&self, conditions: &Options) -> Result<(), Error>;
    fn clone_model_options(&self, options: &Options) -> Result<Options, Error>;
    fn hook_mode(&self, patcher: &Options) -> Result<Options, Error>;
    fn total_hook_groups(&self, conditions: &Options) -> Result<usize, Error>;
    fn min_vram_mode(&self) -> Result<Options, Error>;
    fn set_hook_mode(&self, patcher: &Options, mode: &Options) -> Result<(), Error>;
    fn prepare_model(
        &self,
        patcher: &Options,
        conditions: &Options,
        options: &Options,
    ) -> Result<(), Error>;
    fn filter_hooks(&self, conditions: &Options, options: &Options) -> Result<(), Error>;
    fn outer_method(&self) -> Result<Original<Options>, Error>;
    fn canonical_guider(&self) -> Rc<dyn Any>;
    fn offload_device(&self, patcher: &Options) -> Result<Options, Error>;
    fn cast_offload_options(&self, options: &Options, device: &Options) -> Result<(), Error>;
    fn restore_hook_patches(&self, patcher: &Options) -> Result<(), Error>;
    fn unpack_nested(&self, output: &Options, shapes: &Options) -> Result<Options, Error>;
}
fn none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn assign(slot: &mut Option<Options>, value: Options) {
    drop(slot.replace(value));
}
fn unbound(name: &str) -> Error {
    crate::patching::exception(
        "UnboundLocalError",
        format!("cannot access local variable '{name}' where it is not associated with a value"),
    )
}
#[derive(Default)]
struct Locals {
    ignored: Option<Options>,
    sampler_shapes: Option<Options>,
    packed_callback: Option<Options>,
    masks: Option<Options>,
    key: Option<Options>,
    original_options: Option<Options>,
    original_hook_mode: Option<Options>,
    output: Option<Options>,
    executor: Option<WrapperExecutor<Options>>,
}
pub fn sample(
    state: &GuiderState,
    mut arguments: SampleArguments,
    bindings: &dyn SampleBindings,
) -> Result<Options, Error> {
    if bindings.empty_sigmas(&arguments.sigmas)? {
        return Ok(arguments.latent_image);
    }
    let mut locals = Locals::default();
    if bindings.is_nested(&arguments.latent_image)? {
        let unbound_latents = bindings.unbind(&arguments.latent_image)?;
        let mut items =
            SequenceIterator::for_operation(&unbound_latents, "sampler shape comprehension")?;
        drop(unbound_latents);
        let mut shapes = vec![];
        while let Some(latent) = items.next()? {
            shapes.push(bindings.shape_tuple(&bindings.shape(&latent)?)?);
        }
        drop(items);
        locals.sampler_shapes = Some(list(shapes));
        let values = bindings.unbind(&arguments.latent_image)?;
        let packed = bindings.pack(&values);
        drop(values);
        let [latent, shapes] = packed?;
        arguments.latent_image = latent;
        arguments.latent_shapes = shapes;
        let values = bindings.unbind(&arguments.noise)?;
        let packed = bindings.pack(&values);
        drop(values);
        let [noise, ignored] = packed?;
        arguments.noise = noise;
        assign(&mut locals.ignored, ignored);
    } else {
        arguments.latent_shapes = list(vec![bindings.shape(&arguments.latent_image)?]);
        locals.sampler_shapes = Some(list(vec![
            bindings.shape_tuple(&bindings.shape(&arguments.latent_image)?)?,
        ]));
    }
    bindings.detail(
        &state.fields.get("model_patcher")?,
        locals.sampler_shapes.as_ref().unwrap(),
    )?;
    if bindings.length(&arguments.latent_shapes)? > 1 && !none(&arguments.callback) {
        locals.packed_callback = Some(arguments.callback.clone());
        arguments.callback = bindings.packed_callback(
            locals.packed_callback.as_ref().unwrap(),
            &arguments.latent_shapes,
        )?;
    }
    if !none(&arguments.denoise_mask) {
        if bindings.is_nested(&arguments.denoise_mask)? {
            locals.masks = Some(bindings.unbind(&arguments.denoise_mask)?);
            let masks = bindings.prefix(
                locals.masks.as_ref().unwrap(),
                bindings.length(&arguments.latent_shapes)?,
            )?;
            assign(&mut locals.masks, masks);
        } else {
            locals.masks = Some(list(vec![arguments.denoise_mask.clone()]));
        }
        let start = bindings.length(locals.masks.as_ref().unwrap())?;
        let count = bindings.length(&arguments.latent_shapes)?;
        for i in start..count {
            let mask = bindings.ones(&bindings.index(&arguments.latent_shapes, i)?)?;
            bindings.append(locals.masks.as_ref().unwrap(), mask)?;
        }
        let count = bindings.length(locals.masks.as_ref().unwrap())?;
        for i in 0..count {
            let mask = bindings.index(locals.masks.as_ref().unwrap(), i)?;
            let shape = bindings.index(&arguments.latent_shapes, i)?;
            let device = bindings.load_device(&state.fields.get("model_patcher")?)?;
            let prepared = bindings.prepare_mask(&mask, &shape, &device);
            drop(device);
            drop(shape);
            drop(mask);
            bindings.set_index(locals.masks.as_ref().unwrap(), i, prepared?)?;
        }
        if bindings.length(locals.masks.as_ref().unwrap())? > 1 {
            let [mask, ignored] = bindings.pack(locals.masks.as_ref().unwrap())?;
            arguments.denoise_mask = mask;
            assign(&mut locals.ignored, ignored);
        } else {
            arguments.denoise_mask = bindings.index(locals.masks.as_ref().unwrap(), 0)?;
        }
        arguments.denoise_mask = bindings.mask_float(&arguments.denoise_mask)?;
    }
    state.fields.set("conds", bindings.empty_dictionary()?);
    let mut keys = SequenceIterator::for_operation(
        &state.fields.get("original_conds")?,
        "original condition keys",
    )?;
    while let Some(key) = keys.next()? {
        assign(&mut locals.key, key);
        let records = state
            .fields
            .get("original_conds")?
            .item(locals.key.as_ref().unwrap())?;
        let mut records = SequenceIterator::for_operation(&records, "condition record copies")?;
        let mut copies = vec![];
        while let Some(record) = records.next()? {
            copies.push(bindings.copy_condition_record(&record)?);
        }
        drop(records);
        state
            .fields
            .get("conds")?
            .set_item(locals.key.as_ref().unwrap().clone(), list(copies))?;
    }
    drop(keys);
    bindings.preprocess_hooks(&state.fields.get("conds")?)?;
    let body: Result<(), Error> = (|| {
        locals.original_options = Some(state.fields.get("model_options")?);
        state.fields.set(
            "model_options",
            bindings.clone_model_options(&state.fields.get("model_options")?)?,
        );
        locals.original_hook_mode = Some(bindings.hook_mode(&state.fields.get("model_patcher")?)?);
        if bindings.total_hook_groups(&state.fields.get("conds")?)? <= 1 {
            let mode = bindings.min_vram_mode()?;
            bindings.set_hook_mode(&state.fields.get("model_patcher")?, &mode)?;
        }
        bindings.prepare_model(
            &state.fields.get("model_patcher")?,
            &state.fields.get("conds")?,
            &state.fields.get("model_options")?,
        )?;
        bindings.filter_hooks(
            &state.fields.get("conds")?,
            &state.fields.get("model_options")?,
        )?;
        let original = bindings.outer_method()?;
        let guider = bindings.canonical_guider();
        let wrappers = super::preparation_wrappers::all_wrappers_of(
            "outer_sample",
            &state.fields.get("model_options")?,
        )?;
        locals.executor = Some(WrapperExecutor::new(original, Some(guider), &wrappers, 0));
        drop(wrappers);
        let output = locals.executor.as_ref().unwrap().execute(
            &[
                arguments.noise.clone(),
                arguments.latent_image.clone(),
                arguments.sampler.clone(),
                arguments.sigmas.clone(),
                arguments.denoise_mask.clone(),
                arguments.callback.clone(),
                arguments.disable_pbar.clone(),
                arguments.seed.clone(),
            ],
            &vec![("latent_shapes".into(), arguments.latent_shapes.clone())],
        )?;
        locals.output = Some(output);
        Ok(())
    })();
    crate::source_exception::finally(body, || {
        let options = state.fields.get("model_options")?;
        let device = bindings.offload_device(&state.fields.get("model_patcher")?)?;
        let cast = bindings.cast_offload_options(&options, &device);
        drop(device);
        drop(options);
        cast?;
        let options = locals
            .original_options
            .as_ref()
            .ok_or_else(|| unbound("orig_model_options"))?
            .clone();
        state.fields.set("model_options", options);
        let mode = locals
            .original_hook_mode
            .as_ref()
            .ok_or_else(|| unbound("orig_hook_mode"))?;
        bindings.set_hook_mode(&state.fields.get("model_patcher")?, mode)?;
        bindings.restore_hook_patches(&state.fields.get("model_patcher")?)?;
        Ok(())
    })?;
    state.fields.delete("conds")?;
    if bindings.length(&arguments.latent_shapes)? > 1 {
        let output =
            bindings.unpack_nested(locals.output.as_ref().unwrap(), &arguments.latent_shapes)?;
        assign(&mut locals.output, output);
    }
    Ok(locals.output.as_ref().unwrap().clone())
}
