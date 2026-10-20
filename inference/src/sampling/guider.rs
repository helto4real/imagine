//! Shared CFGGuider outer sampling lifecycle. Model/control/device operations
//! remain required bindings; a failed prepare or context entry does not run
//! cleanup that the original body has not entered yet.
use crate::{
    native::Error,
    patching::{attributes::AttributeStore, options::Options, options_iterator::SequenceIterator},
};

pub struct GuiderState {
    pub fields: AttributeStore,
}
impl Default for GuiderState {
    fn default() -> Self {
        Self {
            fields: AttributeStore::new("CFGGuider"),
        }
    }
}
#[derive(Clone)]
pub struct SampleArguments {
    pub noise: Options,
    pub latent_image: Options,
    pub sampler: Options,
    pub sigmas: Options,
    pub denoise_mask: Options,
    pub callback: Options,
    pub disable_pbar: Options,
    pub seed: Options,
    pub latent_shapes: Options,
}
pub trait DeviceContext {
    fn enter(&self) -> Result<(), Error>;
    /// The frozen cuda_device_context never suppresses a body exception.
    /// A restoration failure still replaces that exception.
    fn exit(&self) -> Result<(), Error>;
}
pub trait OuterBindings {
    fn shape(&self, tensor: &Options) -> Result<Options, Error>;
    fn prepare(
        &self,
        patcher: &Options,
        shape: &Options,
        conds: &Options,
        options: &Options,
    ) -> Result<Options, Error>;
    fn unpack_prepared(&self, prepared: &Options) -> Result<[Options; 3], Error>;
    fn load_device(&self, patcher: &Options) -> Result<Options, Error>;
    fn model_dtype(&self, patcher: &Options) -> Result<Options, Error>;
    fn multigpu_patchers(
        &self,
        patcher: &Options,
        loaded: &Options,
        options: &Options,
    ) -> Result<Options, Error>;
    fn truth(&self, value: &Options) -> Result<bool, Error>;
    fn thread_pool(&self, devices: &Options) -> Result<Options, Error>;
    fn device_context(&self, device: &Options) -> Result<Box<dyn DeviceContext + '_>, Error>;
    fn move_float32(&self, value: &Options, device: &Options) -> Result<Options, Error>;
    fn move_sigmas(&self, value: &Options, device: &Options) -> Result<Options, Error>;
    fn cast_options(
        &self,
        options: &Options,
        device: &Options,
        dtype: &Options,
    ) -> Result<(), Error>;
    fn pre_run(&self, patcher: &Options) -> Result<(), Error>;
    fn inner_sample(&self, arguments: &SampleArguments, device: &Options)
    -> Result<Options, Error>;
    fn shutdown(&self, pool: &Options) -> Result<(), Error>;
    fn cleanup(&self, patcher: &Options) -> Result<(), Error>;
    fn cleanup_models(&self, conditions: &Options, loaded: &Options) -> Result<(), Error>;
}
pub struct ProcessConditions<'a> {
    pub model: &'a Options,
    pub noise: &'a Options,
    pub conditions: &'a Options,
    pub device: &'a Options,
    pub latent: &'a Options,
    pub mask: &'a Options,
    pub seed: &'a Options,
    pub latent_shapes: &'a Options,
}
pub trait InnerBindings {
    fn set_latent_shapes(&self, model: &Options, shapes: &Options) -> Result<(), Error>;
    fn count_nonzero_positive(&self, latent: &Options) -> Result<bool, Error>;
    fn process_latent_in(&self, model: &Options, latent: &Options) -> Result<Options, Error>;
    fn process_conditions(&self, arguments: ProcessConditions<'_>) -> Result<Options, Error>;
    fn clone_model_options(&self, options: &Options) -> Result<Options, Error>;
    fn empty_dictionary(&self) -> Result<Options, Error>;
    fn setdefault(&self, options: &Options, key: &str, default: Options) -> Result<Options, Error>;
    fn extra_arguments(&self, model_options: Options, seed: Options) -> Result<Options, Error>;
    fn sampler_method(&self, sampler: &Options) -> Result<Options, Error>;
    fn sampler_wrappers(&self, model_options: &Options) -> Result<Options, Error>;
    fn executor(
        &self,
        method: &Options,
        sampler: &Options,
        wrappers: &Options,
    ) -> Result<Options, Error>;
    /// Invoke with the canonical guider as first argument, followed by sigmas,
    /// extra_args, callback, noise, latent, mask and disable_pbar.
    fn execute(
        &self,
        executor: &Options,
        arguments: &SampleArguments,
        extra_args: &Options,
    ) -> Result<Options, Error>;
    fn samples_float32(&self, samples: &Options) -> Result<Options, Error>;
    fn latent_out_method(&self, model: &Options) -> Result<Options, Error>;
    fn process_latent_out(&self, method: &Options, samples: &Options) -> Result<Options, Error>;
}
pub fn inner_sample(
    state: &GuiderState,
    mut arguments: SampleArguments,
    device: &Options,
    bindings: &dyn InnerBindings,
) -> Result<Options, Error> {
    bindings.set_latent_shapes(&state.fields.get("inner_model")?, &arguments.latent_shapes)?;
    if !none(&arguments.latent_image) && bindings.count_nonzero_positive(&arguments.latent_image)? {
        arguments.latent_image = bindings
            .process_latent_in(&state.fields.get("inner_model")?, &arguments.latent_image)?;
    }
    let model = state.fields.get("inner_model")?;
    let conditions = state.fields.get("conds")?;
    let processed = bindings.process_conditions(ProcessConditions {
        model: &model,
        noise: &arguments.noise,
        conditions: &conditions,
        device,
        latent: &arguments.latent_image,
        mask: &arguments.denoise_mask,
        seed: &arguments.seed,
        latent_shapes: &arguments.latent_shapes,
    });
    drop(conditions);
    drop(model);
    state.fields.set("conds", processed?);
    let extra_model_options = bindings.clone_model_options(&state.fields.get("model_options")?)?;
    let default = bindings.empty_dictionary()?;
    let transformer = bindings.setdefault(&extra_model_options, "transformer_options", default)?;
    transformer.set("sample_sigmas", arguments.sigmas.clone())?;
    drop(transformer);
    let extra_args =
        bindings.extra_arguments(extra_model_options.clone(), arguments.seed.clone())?;
    // Method lookup precedes wrapper collection and executor construction.
    let method = bindings.sampler_method(&arguments.sampler)?;
    let wrapper_options = extra_args
        .get("model_options")?
        .ok_or_else(|| crate::patching::exception("KeyError", "'model_options'"))?;
    let wrappers = bindings.sampler_wrappers(&wrapper_options);
    drop(wrapper_options);
    let wrappers = wrappers?;
    let executor = bindings.executor(&method, &arguments.sampler, &wrappers);
    drop(wrappers);
    drop(method);
    let executor = executor?;
    let samples = bindings.execute(&executor, &arguments, &extra_args)?;
    // Python evaluates the bound process_latent_out before samples.to(...).
    let model = state.fields.get("inner_model")?;
    let method = bindings.latent_out_method(&model)?;
    drop(model);
    let samples_float = bindings.samples_float32(&samples)?;
    bindings.process_latent_out(&method, &samples_float)
}
#[derive(Default)]
struct Locals {
    extra_devices: Option<Options>,
    all_devices: Option<Options>,
    patcher: Option<Options>,
    output: Option<Options>,
    thread_pool: Option<Options>,
}
fn assign(slot: &mut Option<Options>, value: Options) {
    drop(slot.replace(value));
}
fn list(values: Vec<Options>) -> Options {
    Options::List(std::rc::Rc::new(std::cell::RefCell::new(values)))
}
fn none(value: &Options) -> bool {
    matches!(value, Options::Atom(crate::execution_value::Value::None))
}
pub fn outer_sample(
    state: &GuiderState,
    mut arguments: SampleArguments,
    bindings: &dyn OuterBindings,
) -> Result<Options, Error> {
    let patcher = state.fields.get("model_patcher")?;
    let shape = bindings.shape(&arguments.noise)?;
    let conds = state.fields.get("conds")?;
    let options = state.fields.get("model_options")?;
    let prepared = bindings.prepare(&patcher, &shape, &conds, &options);
    drop(options);
    drop(conds);
    drop(shape);
    drop(patcher);
    let prepared = prepared?;
    let [model, conds, loaded] = bindings.unpack_prepared(&prepared)?;
    drop(prepared);
    state.fields.set("inner_model", model);
    state.fields.set("conds", conds);
    state.fields.set("loaded_models", loaded);
    let device = bindings.load_device(&state.fields.get("model_patcher")?)?;
    let multigpu = bindings.multigpu_patchers(
        &state.fields.get("model_patcher")?,
        &state.fields.get("loaded_models")?,
        &state.fields.get("model_options")?,
    )?;
    let mut locals = Locals::default();
    if bindings.truth(&multigpu)? {
        let mut iter = SequenceIterator::for_operation(&multigpu, "multigpu device comprehension")?;
        let mut devices = vec![];
        while let Some(patcher) = iter.next()? {
            devices.push(bindings.load_device(&patcher)?);
        }
        drop(iter);
        locals.extra_devices = Some(list(devices.clone()));
        devices.insert(0, device.clone());
        locals.all_devices = Some(list(devices));
        let pool = bindings.thread_pool(locals.all_devices.as_ref().unwrap())?;
        state
            .fields
            .get("model_options")?
            .set("multigpu_thread_pool", pool)?;
    }
    let context = bindings.device_context(&device)?;
    context.enter()?;
    let result = {
        let body: Result<(), Error> = (|| {
            arguments.noise = bindings.move_float32(&arguments.noise, &device)?;
            arguments.latent_image = bindings.move_float32(&arguments.latent_image, &device)?;
            arguments.sigmas = bindings.move_sigmas(&arguments.sigmas, &device)?;
            let options = state.fields.get("model_options")?;
            let dtype = bindings.model_dtype(&state.fields.get("model_patcher")?)?;
            let cast = bindings.cast_options(&options, &device, &dtype);
            drop(dtype);
            drop(options);
            cast?;
            bindings.pre_run(&state.fields.get("model_patcher")?)?;
            let mut iter = SequenceIterator::for_operation(&multigpu, "multigpu pre_run")?;
            while let Some(patcher) = iter.next()? {
                assign(&mut locals.patcher, patcher);
                bindings.pre_run(locals.patcher.as_ref().unwrap())?;
            }
            drop(iter);
            assign(
                &mut locals.output,
                bindings.inner_sample(&arguments, &device)?,
            );
            Ok(())
        })();
        crate::source_exception::finally(body, || {
            let pool = state
                .fields
                .get("model_options")?
                .pop("multigpu_thread_pool")?
                .unwrap_or(Options::Atom(crate::execution_value::Value::None));
            assign(&mut locals.thread_pool, pool);
            if !none(locals.thread_pool.as_ref().unwrap()) {
                bindings.shutdown(locals.thread_pool.as_ref().unwrap())?;
            }
            bindings.cleanup(&state.fields.get("model_patcher")?)?;
            let mut iter = SequenceIterator::for_operation(&multigpu, "multigpu cleanup")?;
            while let Some(patcher) = iter.next()? {
                assign(&mut locals.patcher, patcher);
                bindings.cleanup(locals.patcher.as_ref().unwrap())?;
            }
            Ok(())
        })
    };
    crate::source_exception::finally(result, || context.exit())?;
    bindings.cleanup_models(
        &state.fields.get("conds")?,
        &state.fields.get("loaded_models")?,
    )?;
    state.fields.delete("inner_model")?;
    state.fields.delete("loaded_models")?;
    Ok(locals
        .output
        .as_ref()
        .expect("completed inner sample")
        .clone())
}
