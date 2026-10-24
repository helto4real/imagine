//! Model files a Scene's settings name, checked against the H3 model root
//! before admission, so a missing file is named instead of failing in the worker.
use super::{contract::*, planning};
use crate::h3_protocol as h3;
use std::path::Path;

/// Files the effective H3 settings and LoRAs name, as written there. Blank
/// selections use the deployment's own files and are not listed.
pub fn selected(project: &VideoProject, scene: &Scene) -> Vec<String> {
    let mut files: Vec<String> = vec![];
    if project.effective_model(scene).adapter != "h3" {
        return files;
    }
    if let Some(settings) = project
        .effective_settings(scene)
        .and_then(|slot| serde_json::from_value::<planning::H3Settings>(slot.values.clone()).ok())
    {
        let c = &settings.components;
        files.extend(
            [&c.model, &c.text_encoder, &c.video_vae, &c.audio_vae]
                .into_iter()
                .flatten()
                .cloned(),
        );
        if let Some(second) = &settings.stage_two {
            let own = if second.model.variant == "i2v" {
                &second.i2v_model_file
            } else {
                &second.reference_model_file
            };
            files.extend(own.clone().filter(|f| !f.trim().is_empty()).or_else(|| {
                planning::inherited_stage_two_model(project, scene, &settings)
                    .ok()
                    .flatten()
            }));
            if second.handoff_upscale_factor > 1.
                && second.handoff == planning::H3Handoff::ReNoiseVideoPrediction
            {
                files.extend(second.trained_latent_upscale_model.clone());
            }
        }
    }
    let loras = imagine_video_contract::loras::resolve(
        &project.creative.h3_loras.clone().unwrap_or_default(),
        project.creative.scene_loras.get(&scene.id),
    );
    files.extend(
        loras
            .active()
            .into_iter()
            .map(|row| format!("loras/{}", row.name)),
    );
    files.retain(|f| !f.trim().is_empty());
    let mut seen = std::collections::BTreeSet::new();
    files.retain(|f| seen.insert(f.clone()));
    files
}

/// The `selected` files that are not files under `root`, as the worker resolves them.
pub fn missing(project: &VideoProject, scene: &Scene, root: &Path) -> Vec<String> {
    selected(project, scene)
        .into_iter()
        .filter(|f| !h3::options::Components::path(root, Some(f), "").is_file())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use imagine_video_contract::generation::{Defaults, h3_model};
    fn project(variant: &str) -> VideoProject {
        let mut p = VideoProject::create(CreateProject {
            schema_version: 1,
            generation: Some(Defaults::default()),
            default_model: h3_model(variant),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap();
        p.scenes.push(Scene {
            id: SceneId("own-scene".into()),
            start: Time { num: 0, den: 1 },
            duration: Time { num: 2, den: 1 },
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
        p
    }
    fn settings(p: &mut VideoProject, variant: &str, edit: impl FnOnce(&mut planning::H3Settings)) {
        let slot = p
            .generation
            .as_mut()
            .unwrap()
            .settings
            .iter_mut()
            .find(|s| s.model.variant == variant)
            .unwrap();
        let mut value: planning::H3Settings = serde_json::from_value(slot.values.clone()).unwrap();
        edit(&mut value);
        slot.values = serde_json::to_value(value).unwrap();
    }
    #[test]
    fn named_files_lose_blanks_and_repeats_and_missing_ones_are_reported() {
        let root = tempfile::tempdir().unwrap();
        let mut p = project("i2v");
        settings(&mut p, "i2v", |s| {
            s.components.text_encoder = Some("  ".into());
            s.components.video_vae = Some("vae/own.safetensors".into());
        });
        p.creative.h3_loras = Some(imagine_video_contract::loras::Stack {
            loras: vec![imagine_video_contract::loras::Row {
                enabled: true,
                name: "own-style.safetensors".into(),
                strength_model: 1.,
                strength_clip: 1.,
            }],
            ..Default::default()
        });
        let scene = p.scenes[0].clone();
        let model = "diffusion_models/minimax_h3/minimaxH3.int8.safetensors";
        assert_eq!(
            selected(&p, &scene),
            [model, "vae/own.safetensors", "loras/own-style.safetensors"]
        );
        assert_eq!(missing(&p, &scene, root.path()).len(), 3);
        for file in [model, "loras/own-style.safetensors"] {
            let path = root.path().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"own").unwrap();
        }
        assert_eq!(missing(&p, &scene, root.path()), ["vae/own.safetensors"]);
        // An absolute selection is checked where it points.
        let absolute = root.path().join("vae/own.safetensors");
        settings(&mut p, "i2v", |s| {
            s.components.video_vae = Some(absolute.to_string_lossy().into_owned());
        });
        assert_eq!(missing(&p, &scene, root.path()).len(), 1);
        std::fs::create_dir_all(absolute.parent().unwrap()).unwrap();
        std::fs::write(&absolute, b"own").unwrap();
        assert!(missing(&p, &scene, root.path()).is_empty());
    }
    #[test]
    fn stage_two_names_its_own_or_inherited_model_and_the_trained_upscaler() {
        let mut p = project("i2v");
        settings(&mut p, "i2v", |s| {
            s.stage_two = Some(planning::H3SecondStage {
                handoff_upscale_factor: 2.,
                reference_model_file: None,
                i2v_model_file: None,
                sampling: s.stage_one.clone(),
                model: h3_model("reference"),
                handoff: planning::H3Handoff::ReNoiseVideoPrediction,
                trained_latent_upscale_model: Some("latent_upscale_models/own.safetensors".into()),
            });
        });
        let scene = p.scenes[0].clone();
        assert_eq!(
            selected(&p, &scene),
            [
                "diffusion_models/minimax_h3/minimaxH3.int8.safetensors",
                // The Reference mode's model, inherited by Stage 2.
                "diffusion_models/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "latent_upscale_models/own.safetensors",
            ]
        );
    }
}
