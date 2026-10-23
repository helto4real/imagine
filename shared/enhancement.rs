//! Prompt enhancement wire types, independent of image inference.
use serde::{Deserialize, Serialize};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:5002/v1";
pub const DEFAULT_SYSTEM_PROMPT: &str = "Improve the user's image-generation prompt for clarity and useful visual detail. Preserve their intent, subjects, reference-image bindings, language, and explicit constraints. Do not invent additional characters or change requested actions. Return only the improved prompt, without commentary, headings, or quotation marks.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoryTransport {
    #[default]
    Koboldcpp,
    Ollama,
}
impl StoryTransport {
    pub fn name(self) -> &'static str {
        match self {
            Self::Koboldcpp => "koboldcpp",
            Self::Ollama => "ollama",
        }
    }
    pub fn is_default(&self) -> bool {
        *self == Self::Koboldcpp
    }
}

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
#[serde(remote = "Self", deny_unknown_fields)]
pub struct Profile {
    #[serde(default, skip_serializing_if = "StoryTransport::is_default")]
    pub story_transport: StoryTransport,
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
}
impl Default for Profile {
    fn default() -> Self {
        Self {
            story_transport: StoryTransport::default(),
            id: "proxy".into(),
            name: "Proxy".into(),
            base_url: DEFAULT_BASE_URL.into(),
            model: "koboldcpp/initial_model".into(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            max_tokens: 1024,
            token_parameter: TokenParameter::default(),
            temperature: None,
            parameters: Default::default(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ProfileUpdate {
    pub profile: Profile,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ModelsRequest {
    pub profile_id: String,
    pub base_url: String,
}

// Read only the former credential field names for compatibility. Their values
// are discarded before constructing the current type and can never be sent or
// serialized again. Remote derives retain each type's ordinary field validation.
macro_rules! without_legacy_credentials {
    ($name:ident, $($legacy:literal),+ $(,)?) => {
        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                $name::serialize(self, serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let mut value = serde_json::Value::deserialize(deserializer)?;
                if let Some(fields) = value.as_object_mut() {
                    $(fields.remove($legacy);)+
                }
                $name::deserialize(value).map_err(serde::de::Error::custom)
            }
        }
    };
}
without_legacy_credentials!(Profile, "has_api_key", "api_key");
without_legacy_credentials!(ProfileUpdate, "api_key");
without_legacy_credentials!(ModelsRequest, "api_key");

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
