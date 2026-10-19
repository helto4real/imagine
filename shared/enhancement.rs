//! Prompt enhancement wire types, independent of image inference.
use serde::{Deserialize, Serialize};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:5002/v1";
pub const DEFAULT_SYSTEM_PROMPT: &str = "Improve the user's image-generation prompt for clarity and useful visual detail. Preserve their intent, subjects, reference-image bindings, language, and explicit constraints. Do not invent additional characters or change requested actions. Return only the improved prompt, without commentary, headings, or quotation marks.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenParameter {
    #[default]
    MaxTokens,
    MaxCompletionTokens,
}
impl TokenParameter {
    pub fn name(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub model: String,
    pub system_prompt: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub token_parameter: TokenParameter,
    pub temperature: Option<f32>,
    /// Optional provider parameters; omitted values retain server defaults.
    #[serde(default)]
    pub parameters: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub has_api_key: bool,
}
impl Default for Profile {
    fn default() -> Self {
        Self {
            id: "proxy".into(),
            name: "Proxy".into(),
            base_url: DEFAULT_BASE_URL.into(),
            model: "koboldcpp/initial_model".into(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            max_tokens: 1024,
            token_parameter: TokenParameter::default(),
            temperature: None,
            parameters: Default::default(),
            has_api_key: false,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileUpdate {
    pub profile: Profile,
    /// None preserves the saved key; an empty string clears it.
    pub api_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsRequest {
    pub profile_id: String,
    pub base_url: String,
    /// None uses the saved key only if the saved server address still matches.
    pub api_key: Option<String>,
}

/// These fields are controlled by the enhancement operation or dedicated controls.
pub fn reserved_parameter(name: &str) -> bool {
    matches!(
        name,
        "model"
            | "messages"
            | "stream"
            | "n"
            | "max_tokens"
            | "max_completion_tokens"
            | "temperature"
    )
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: String,
    pub profile_id: String,
    pub prompt: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Status {
    Running,
    Succeeded { text: String },
    Failed { code: String },
    Cancelled,
}
