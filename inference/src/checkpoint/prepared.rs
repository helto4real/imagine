//! Source-named parameters plus Comfy module policy. Policy is not encoded as
//! extra checkpoint keys or inferred from a model's filename.
use super::{LoadOptions, QuantizedModule, load_quantized_module};
use crate::{
    execution_value::Value,
    native::{Error, Kitchen},
};
use std::collections::BTreeMap;
use tch::Tensor;

#[derive(Clone, Default)]
pub struct LinearPolicy {
    pub full_precision: bool,
    pub full_precision_config: bool,
    pub force_forward: bool,
    pub pre_quant_scale: Option<Value>,
    pub input_scale: Option<Value>,
}
#[derive(Default, Clone)]
pub struct PreparedParameters {
    pub values: BTreeMap<String, Value>,
    pub linear_policies: BTreeMap<String, LinearPolicy>,
    pub lowvram_patches: BTreeMap<String, Vec<crate::patching::Patch>>,
}
impl std::ops::Deref for PreparedParameters {
    type Target = BTreeMap<String, Value>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
impl PreparedParameters {
    pub fn plain(values: BTreeMap<String, Value>) -> Self {
        Self {
            values,
            linear_policies: BTreeMap::new(),
            lowvram_patches: BTreeMap::new(),
        }
    }
    /// Consume the source loader's result. Biases/norms and other super-loader
    /// parameters remain explicit, so unsupported/missing keys cannot be filled
    /// by invented values.
    pub fn insert_linear(&mut self, name: &str, module: QuantizedModule) {
        if let Some(weight) = module.weight {
            self.values.insert(format!("{name}.weight"), weight);
        }
        let mut extras = module.extras;
        self.linear_policies.insert(
            name.into(),
            LinearPolicy {
                full_precision: module.full_precision,
                full_precision_config: module.full_precision_config,
                force_forward: false,
                pre_quant_scale: extras.remove("pre_quant_scale").map(Value::tensor),
                input_scale: extras.remove("input_scale").map(Value::tensor),
            },
        );
    }
    pub fn prepare_linear(
        &mut self,
        state: &mut BTreeMap<String, Tensor>,
        name: &str,
        options: LoadOptions<'_>,
        native: &Kitchen,
    ) -> Result<(), Error> {
        let module = load_quantized_module(state, &format!("{name}."), options, native)?;
        self.insert_linear(name, module);
        Ok(())
    }
}
