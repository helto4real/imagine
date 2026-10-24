//! Project defaults and per-model settings; runtime support stays in the adapter.
use crate::{
    contract::*,
    planning::{H3Settings, H3Stage},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
/// ComfyUI's numeric widget control. This affects the next widget value only;
/// the explicitly entered seed retains the full u64 range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SeedControl {
    Fixed,
    Increment,
    Decrement,
    Randomize,
}
impl SeedControl {
    pub fn next(self, seed: u64, random_bits: u64) -> u64 {
        const WIDGET_MAX: u64 = 1 << 50;
        match self {
            Self::Fixed => seed,
            Self::Increment => seed.saturating_add(1).min(WIDGET_MAX),
            Self::Decrement => seed.saturating_sub(1).min(WIDGET_MAX),
            Self::Randomize => random_bits & (WIDGET_MAX - 1),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Canvas {
    pub ratio: String,
    pub portrait: bool,
    pub quality: String,
}
impl Default for Canvas {
    fn default() -> Self {
        Self {
            ratio: "16:9".into(),
            portrait: false,
            quality: "Standard".into(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Defaults {
    pub canvas: Option<Canvas>,
    #[serde(default)]
    pub settings: Vec<SettingsSlot>,
    pub seed: u64,
    #[serde(default)]
    pub random_seed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_after_generate: Option<SeedControl>,
}
pub fn h3_model(variant: &str) -> ModelSelection {
    ModelSelection {
        adapter: "h3".into(),
        model: "minimax-h3".into(),
        variant: variant.into(),
        profile: if variant == "i2v" {
            "h3-fl2va-i2va-v1"
        } else {
            "h3-av-noref-v2"
        }
        .into(),
        profile_revision: "2".into(),
    }
}
pub fn h3_defaults() -> H3Settings {
    H3Settings {
        components: Default::default(),
        shift_video: 12.,
        shift_audio: 3.,
        audio_reference_mode: "Timeline Audio References".into(),
        ref_image_size: "Standard".into(),
        stage_one: H3Stage {
            steps: 20,
            sampler: "res_multistep".into(),
            scheduler: "simple".into(),
            denoise: 1.,
            sigmas: vec![],
        },
        stage_two: None,
        stage_two_draft: None,
        base_steps_to_run: 4,
    }
}
pub fn h3_defaults_for(variant: &str) -> H3Settings {
    let mut settings = h3_defaults();
    settings.components.model = Some(format!(
        "diffusion_models/minimax_h3/{}",
        // The locked I2V file; Director's fl2va name is not installed here.
        if variant == "i2v" {
            "minimaxH3.int8.safetensors"
        } else {
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        }
    ));
    settings
}
impl Default for Defaults {
    fn default() -> Self {
        Self {
            canvas: Some(Canvas::default()),
            settings: ["reference", "i2v"]
                .map(|variant| SettingsSlot {
                    model: h3_model(variant),
                    schema_version: 1,
                    values: serde_json::to_value(h3_defaults_for(variant))
                        .expect("finite defaults"),
                })
                .to_vec(),
            seed: 0,
            random_seed: false,
            control_after_generate: Some(SeedControl::Randomize),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn widget_control_matches_frontend_boundaries_without_restricting_manual_seed() {
        let max = 1 << 50;
        for (mode, seed, random, expected) in [
            (SeedControl::Fixed, u64::MAX, 0, u64::MAX),
            (SeedControl::Increment, 0, 0, 1),
            (SeedControl::Increment, u64::MAX, 0, max),
            (SeedControl::Decrement, 0, 0, 0),
            (SeedControl::Decrement, 42, 0, 41),
            (SeedControl::Decrement, u64::MAX, 0, max),
            (SeedControl::Randomize, 42, u64::MAX, max - 1),
        ] {
            assert_eq!(mode.next(seed, random), expected);
        }
        let mut value = serde_json::to_value(Defaults::default()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("control_after_generate");
        let legacy: Defaults = serde_json::from_value(value).unwrap();
        assert_eq!(legacy.control_after_generate, None);
    }
    #[test]
    fn project_settings_inherit_by_exact_model_and_fingerprint_only_effective_values() {
        let g = Defaults::default();
        assert_eq!(g.canvas.as_ref().unwrap().quality, "Standard");
        assert_eq!(g.seed, 0);
        let mut p = VideoProject::create(CreateProject {
            generation: None,
            schema_version: 1,
            default_model: h3_model("reference"),
            delivery: DeliverySettings {
                width: 768,
                height: 512,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap();
        let t = |n| Time { num: n, den: 1 };
        let id = SceneId("own".into());
        p.scenes.push(Scene {
            id: id.clone(),
            start: t(0),
            duration: t(5),
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![g.settings[0].clone()],
            sections: vec![Section {
                id: SectionId("section".into()),
                start: t(0),
                duration: t(5),
                prompt: String::new(),
            }],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
        let original = crate::planning::input_revision(&p, &id).unwrap();
        p.generation = Some(g);
        p.scenes[0].settings.clear();
        assert_eq!(crate::planning::input_revision(&p, &id).unwrap(), original);
        p.generation.as_mut().unwrap().settings[1].values["stage_one"]["steps"] =
            serde_json::json!(7);
        assert_eq!(crate::planning::input_revision(&p, &id).unwrap(), original);
        p.generation.as_mut().unwrap().settings[0].values["stage_one"]["steps"] =
            serde_json::json!(8);
        assert_ne!(crate::planning::input_revision(&p, &id).unwrap(), original);
        let before_draft = crate::planning::input_revision(&p, &id).unwrap();
        p.generation.as_mut().unwrap().settings[0].values["stage_two_draft"] =
            serde_json::json!({"settings":{"own":"inactive fixture"},"raw_sigmas":"unfinished"});
        assert_eq!(
            crate::planning::input_revision(&p, &id).unwrap(),
            before_draft
        );
        p.generation.as_mut().unwrap().settings[0]
            .values
            .as_object_mut()
            .unwrap()
            .remove("stage_two_draft");
        let own = SettingsSlot {
            model: h3_model("reference"),
            schema_version: 1,
            values: serde_json::to_value(h3_defaults_for("reference")).unwrap(),
        };
        p.scenes[0].settings.push(own);
        assert_eq!(crate::planning::input_revision(&p, &id).unwrap(), original);
        p.scenes[0].model = ModelChoice::Override(h3_model("i2v"));
        assert_eq!(
            p.effective_settings(&p.scenes[0]).unwrap().values["stage_one"]["steps"],
            7
        );
        let value = serde_json::to_value(&p).unwrap();
        assert_eq!(serde_json::from_value::<VideoProject>(value).unwrap(), p);
        p.scenes[0].model = ModelChoice::ProjectDefault;
        p.scenes[0].settings[0].values["stage_two"] = serde_json::json!({
            "model":h3_model("i2v"),"handoff":"continue_latent","trained_latent_upscale_model":null,
            "sampling":{"steps":4,"sampler":"euler","scheduler":"simple"}
        });
        let inherited = crate::planning::input_revision(&p, &id).unwrap();
        p.generation.as_mut().unwrap().settings[1].values["stage_one"]["steps"] =
            serde_json::json!(99);
        assert_eq!(crate::planning::input_revision(&p, &id).unwrap(), inherited);
        p.generation.as_mut().unwrap().settings[1].values["components"]["model"] =
            serde_json::json!("any-compatible-alternative.bin");
        assert_ne!(crate::planning::input_revision(&p, &id).unwrap(), inherited);
        p.scenes[0].settings[0].values["stage_two"]["i2v_model_file"] =
            serde_json::json!("explicit-stage2.bin");
        let explicit = crate::planning::input_revision(&p, &id).unwrap();
        p.generation.as_mut().unwrap().settings[1].values["components"]["model"] =
            serde_json::json!("ignored-alternative.bin");
        assert_eq!(crate::planning::input_revision(&p, &id).unwrap(), explicit);
        use crate::loras::{Merge, Override, Row, Stack};
        p.creative.h3_loras = Some(Stack {
            loras: vec![Row {
                name: "/own/alternative.weights".into(),
                strength_model: 0.5,
                ..Default::default()
            }],
            ..Default::default()
        });
        let patched = crate::planning::input_revision(&p, &id).unwrap();
        assert_ne!(patched, explicit);
        p.creative.scene_loras.insert(
            id.clone(),
            Override {
                enabled: true,
                merge_mode: Merge::Disable,
                ..Default::default()
            },
        );
        assert_eq!(crate::planning::input_revision(&p, &id).unwrap(), explicit);
        p.creative.scene_loras.get_mut(&id).unwrap().enabled = false;
        assert_eq!(crate::planning::input_revision(&p, &id).unwrap(), patched);
    }
}
