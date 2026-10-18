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
    /// ComfyAttention stores decoded Python config, not a registered Tensor.
    pub attention_configs: BTreeMap<String, serde_json::Value>,
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
            attention_configs: BTreeMap::new(),
            linear_policies: BTreeMap::new(),
            lowvram_patches: BTreeMap::new(),
        }
    }
    pub fn attention_config(&self, name: &str) -> Result<Option<serde_json::Value>, Error> {
        if let Some(config) = self.attention_configs.get(name) {
            return Ok(Some(config.clone()));
        }
        self.values
            .get(name)
            .map(Self::decode_attention_config)
            .transpose()
    }
    fn decode_attention_config(value: &Value) -> Result<serde_json::Value, Error> {
        let metadata = value.as_tensor()?;
        if metadata.device() != tch::Device::Cpu {
            return Err(Error::ReferenceException {class:"TypeError",message:"can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.".into()});
        }
        let bytes = metadata
            .f_contiguous()?
            .f_view_dtype(tch::Kind::Uint8)?
            .f_reshape([-1])?;
        let mut data = vec![0; bytes.numel()];
        let length = data.len();
        bytes.f_copy_data_u8(&mut data, length)?;
        serde_json::from_slice(&data)
            .map_err(|e| Error::Native(format!("Invalid ComfyAttention config: {e}")))
    }
    pub fn insert_attention_config(&mut self, name: String, value: Value) -> Result<(), Error> {
        self.attention_configs
            .insert(name, Self::decode_attention_config(&value)?);
        Ok(())
    }
    /// module_size uses ComfyAttention._save_to_state_dict's fresh JSON bytes.
    pub fn attention_state_dict_bytes(&self) -> u64 {
        self.attention_configs.values().map(json_bytes).sum()
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
fn json_string_bytes(text: &str) -> u64 {
    2 + text
        .chars()
        .map(|c| match c {
            '"' | '\\' | '\u{8}' | '\u{c}' | '\n' | '\r' | '\t' => 2,
            c if !(' '..'\u{7f}').contains(&c) => 6 * c.len_utf16() as u64,
            _ => 1,
        })
        .sum::<u64>()
}
fn json_bytes(value: &serde_json::Value) -> u64 {
    use serde_json::Value as J;
    match value {
        J::Null => 4,
        J::Bool(true) => 4,
        J::Bool(false) => 5,
        J::String(s) => json_string_bytes(s),
        J::Number(n) => {
            if n.is_f64() {
                crate::registry::python_float(n.as_f64().unwrap()).len() as u64
            } else {
                n.to_string().len() as u64
            }
        }
        J::Array(values) => {
            2 + values.iter().map(json_bytes).sum::<u64>()
                + 2 * values.len().saturating_sub(1) as u64
        }
        J::Object(values) => {
            2 + values
                .iter()
                .map(|(key, value)| json_string_bytes(key) + 2 + json_bytes(value))
                .sum::<u64>()
                + 2 * values.len().saturating_sub(1) as u64
        }
    }
}
