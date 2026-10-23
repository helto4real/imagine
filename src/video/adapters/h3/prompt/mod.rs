//! Pure Director H3 prompt compilers. Media conditioning is a separate runtime contract.
mod common;
pub mod dialogue;
mod i2v;
mod manifest_text;
mod reference;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Input {
    pub sections: Vec<Value>,
    pub reference_manifest: Value,
    pub global_prompt: Value,
    pub scene_prompt: Value,
    pub text_references: Vec<Value>,
    pub keyframes: Value,
    pub effective_duration_seconds: f64,
    pub boundary_prompt: String,
}
#[derive(Serialize)]
pub struct Output {
    pub prompt: String,
    pub metadata: Value,
}
pub fn compile_i2v(input: &Input) -> Output {
    i2v::compile(input)
}
pub fn compile_reference(input: &Input) -> Output {
    reference::compile(input)
}
fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap_or("")
}
fn f(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or(0.)
}
fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
