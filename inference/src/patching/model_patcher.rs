//! Shared ModelPatcher instance state and original constructor body. Native
//! model attributes, UUID/set/enum allocation retain required typed bindings.
use super::{attributes::AttributeStore, options::Options, relationships::ModelRelationships};
use crate::{execution_value::Value as V, native::Error};

pub struct PatcherState {
    pub fields: AttributeStore,
    pub relationships: ModelRelationships,
    pub registries: super::model_registry::ModelRegistries,
}
pub struct Initialization {
    pub model: Options,
    pub load_device: Options,
    pub offload_device: Options,
    pub size: Options,
    pub weight_inplace_update: Options,
    pub fast_disk: Options,
}
pub trait InitializationBindings {
    fn model_attribute(&self, model: &Options, name: &str) -> Result<Options, Error>;
    fn set_model_attribute(&self, model: &Options, name: &str, value: Options)
    -> Result<(), Error>;
    fn debug(&self, message: &str) -> Result<(), Error>;
    fn uuid4(&self) -> Result<Options, Error>;
    fn empty_pinned_set(&self) -> Result<Options, Error>;
    fn max_speed_hook_mode(&self) -> Result<Options, Error>;
    fn empty_dictionary(&self) -> Result<Options, Error>;
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn boolean(value: bool) -> Options {
    Options::Atom(V::Bool(value))
}
impl PatcherState {
    /// Allocates support slots, not source fields/containers or a second model
    /// owner. This state is intended to belong to the canonical patcher handle.
    pub fn uninitialized(type_name: &str) -> Self {
        let fields = AttributeStore::new(type_name);
        let relationships = ModelRelationships::bind(&fields);
        let registries = super::model_registry::ModelRegistries::bind(&fields);
        Self {
            fields,
            relationships,
            registries,
        }
    }
    pub fn initialize(
        &self,
        args: &Initialization,
        bindings: &dyn InitializationBindings,
    ) -> Result<(), Error> {
        let fields = &self.fields;
        fields.set("size", args.size.clone());
        fields.set("model", args.model.clone());
        fields.set("fast_disk", args.fast_disk.clone());
        if !self.has_model_attribute("device", bindings)? {
            bindings.debug("Model doesn't have a device attribute.")?;
            let model = fields.get("model")?;
            bindings.set_model_attribute(&model, "device", args.offload_device.clone())?;
        } else {
            let model = fields.get("model")?;
            let device = bindings.model_attribute(&model, "device");
            drop(model);
            let device = device?;
            let is_none = matches!(device, Options::Atom(V::None));
            drop(device);
            if is_none {
                let model = fields.get("model")?;
                bindings.set_model_attribute(&model, "device", args.offload_device.clone())?;
            }
        }
        for name in [
            "patches",
            "backup",
            "backup_buffers",
            "object_patches",
            "object_patches_backup",
            "weight_wrapper_patches",
        ] {
            fields.set(name, bindings.empty_dictionary()?);
        }
        let transformer_options = bindings.empty_dictionary()?;
        let model_options = bindings.empty_dictionary()?;
        model_options.set("transformer_options", transformer_options)?;
        fields.set("model_options", model_options);
        fields.set("load_device", args.load_device.clone());
        fields.set("offload_device", args.offload_device.clone());
        fields.set("weight_inplace_update", args.weight_inplace_update.clone());
        fields.set("force_cast_weights", boolean(false));
        fields.set("patches_uuid", bindings.uuid4()?);
        fields.set("parent", none());
        fields.set("pinned", bindings.empty_pinned_set()?);
        for name in ["attachments", "additional_models", "callbacks", "wrappers"] {
            fields.set(name, bindings.empty_dictionary()?);
        }
        fields.set("is_injected", boolean(false));
        fields.set("skip_injection", boolean(false));
        fields.set("injections", bindings.empty_dictionary()?);
        fields.set("hook_patches", bindings.empty_dictionary()?);
        fields.set("hook_patches_backup", none());
        fields.set("hook_backup", bindings.empty_dictionary()?);
        fields.set("cached_hook_patches", bindings.empty_dictionary()?);
        fields.set("current_hooks", none());
        fields.set("forced_hooks", none());
        fields.set("is_clip", boolean(false));
        fields.set("hook_mode", bindings.max_speed_hook_mode()?);
        fields.set("cached_patcher_init", none());
        fields.set("is_multigpu_base_clone", boolean(false));
        fields.set("clone_base_uuid", bindings.uuid4()?);
        for (name, value) in [
            ("model_loaded_weight_memory", Options::Atom(V::Int(0))),
            ("lowvram_patch_counter", Options::Atom(V::Int(0))),
            ("model_lowvram", boolean(false)),
            ("current_weight_patches_uuid", none()),
            ("model_offload_buffer_memory", Options::Atom(V::Int(0))),
        ] {
            if !self.has_model_attribute(name, bindings)? {
                let model = fields.get("model")?;
                bindings.set_model_attribute(&model, name, value)?;
            }
        }
        Ok(())
    }
    fn has_model_attribute(
        &self,
        name: &str,
        bindings: &dyn InitializationBindings,
    ) -> Result<bool, Error> {
        let model = self.fields.get("model")?;
        let attribute = bindings.model_attribute(&model, name);
        // hasattr releases the result while its model argument is still alive.
        let outcome = match attribute {
            Ok(value) => {
                drop(value);
                Ok(true)
            }
            Err(caught) if caught.is_reference("AttributeError") => Ok(false),
            Err(error) => Err(error),
        };
        drop(model);
        outcome
    }
}
