//! Director's H3 main-target stack resolution. Row order and editor state survive.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
fn one() -> f64 {
    1.
}
fn enabled() -> bool {
    true
}
fn single() -> String {
    "single".into()
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(from = "RowInput")]
pub struct Row {
    #[serde(default = "enabled")]
    pub enabled: bool,
    pub name: String,
    #[serde(default = "one")]
    pub strength_model: f64,
    #[serde(default = "one")]
    pub strength_clip: f64,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RowInput {
    #[serde(default = "enabled")]
    enabled: bool,
    name: String,
    #[serde(default = "one")]
    strength_model: f64,
    #[serde(default)]
    strength_clip: Option<f64>,
}
impl From<RowInput> for Row {
    fn from(row: RowInput) -> Self {
        Self {
            enabled: row.enabled,
            name: row.name,
            strength_model: row.strength_model,
            strength_clip: row.strength_clip.unwrap_or(row.strength_model),
        }
    }
}
impl Default for Row {
    fn default() -> Self {
        Self {
            enabled: true,
            name: String::new(),
            strength_model: 1.,
            strength_clip: 1.,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Ui {
    pub show_strengths: String,
    pub r#match: String,
}
impl Default for Ui {
    fn default() -> Self {
        Self {
            show_strengths: single(),
            r#match: String::new(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Stack {
    pub version: u32,
    pub loras: Vec<Row>,
    pub ui: Ui,
}
impl Default for Stack {
    fn default() -> Self {
        Self {
            version: 1,
            loras: vec![],
            ui: Ui::default(),
        }
    }
}
impl Stack {
    pub fn active(&self) -> Vec<Row> {
        self.loras
            .iter()
            .filter(|r| {
                r.enabled
                    && !r.name.is_empty()
                    && r.name != "None"
                    && (r.strength_model != 0. || r.strength_clip != 0.)
            })
            .cloned()
            .collect()
    }
    pub fn valid(&self) -> bool {
        self.loras
            .iter()
            .all(|r| r.strength_model.is_finite() && r.strength_clip.is_finite())
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum Merge {
    #[default]
    #[serde(rename = "Inherit Global")]
    Inherit,
    #[serde(rename = "Add To Global")]
    Add,
    #[serde(rename = "Replace Global")]
    Replace,
    #[serde(rename = "Disable LoRAs")]
    Disable,
}
impl Merge {
    pub const ALL: [Self; 4] = [Self::Inherit, Self::Add, Self::Replace, Self::Disable];
    pub fn label(self) -> &'static str {
        match self {
            Self::Inherit => "Inherit Global",
            Self::Add => "Add To Global",
            Self::Replace => "Replace Global",
            Self::Disable => "Disable LoRAs",
        }
    }
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Override {
    pub enabled: bool,
    pub merge_mode: Merge,
    pub main: Stack,
}
pub fn resolve(global: &Stack, scene: Option<&Override>) -> Stack {
    let Some(scene) = scene.filter(|s| s.enabled) else {
        return global.clone();
    };
    match scene.merge_mode {
        Merge::Inherit => global.clone(),
        Merge::Disable => Stack::default(),
        Merge::Replace => scene.main.clone(),
        Merge::Add => {
            let mut stack = global.clone();
            stack.loras.extend(scene.main.loras.clone());
            stack.ui = scene.main.ui.clone();
            stack
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn main_target_resolution_and_active_rows_match_director_without_name_gates() {
        let row: Row =
            serde_json::from_str(r#"{"name":"/own/compatible.weights","strength_model":-0.75}"#)
                .unwrap();
        assert_eq!(row.strength_clip, -0.75); // Director defaults CLIP strength to model strength.
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/director-h3-loras.json")).unwrap();
        for case in fixtures["cases"].as_array().unwrap() {
            let global: Stack = serde_json::from_value(case["global_stack"].clone()).unwrap();
            let scene: Override = serde_json::from_value(case["scene"].clone()).unwrap();
            let actual = resolve(&global, Some(&scene));
            assert_eq!(
                actual,
                serde_json::from_value::<Stack>(case["resolved"].clone()).unwrap()
            );
            assert_eq!(
                actual.active(),
                serde_json::from_value::<Vec<Row>>(case["active"].clone()).unwrap()
            );
        }
    }
}
