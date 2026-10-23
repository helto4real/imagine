//! Admission-time project snapshots; no file paths or image payloads in public drafts.
use super::engine::*;
use crate::{
    enhancement::ProviderImage,
    video::{contract::*, persistence::*},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::io::{Read, Seek};
#[derive(Clone, Default)]
pub(crate) struct Grounding {
    pub images: Vec<(StoryImage, ProviderImage)>,
    pub cast: Value,
}
impl Grounding {
    pub fn capture(session: &ProjectSession, request: &Request) -> Result<Self, &'static str> {
        let base = &session.document().project;
        let mut cast = vec![];
        let mut requested_cast = request.cast.clone();
        let mut selections = request.images.clone();
        if request.operation != Operation::Outline {
            let scene = base
                .scene(request.scene_id.as_ref().ok_or("scene_not_found")?)
                .map_err(|_| "scene_not_found")?;
            let source = base.creative.scene_prompts.get(&scene.id);
            let text = std::iter::once(scene.description.as_str())
                .chain(scene.sections.iter().map(|s| s.prompt.as_str()))
                .chain(std::iter::once(base.global_prompt.as_str()))
                .chain(source.into_iter().flat_map(|s| {
                    [
                        s.summary.as_str(),
                        s.ambient_sound.as_str(),
                        s.non_diegetic_music.as_str(),
                    ]
                }))
                .collect::<Vec<_>>()
                .join("\n");
            let aliases = imagine_video_contract::references::aliases(&text);
            for reference in base.creative.references.iter().filter(|r| {
                r.enabled && aliases.contains(&r.alias.trim_start_matches('@').to_ascii_lowercase())
            }) {
                if !requested_cast.contains(&reference.id) {
                    requested_cast.push(reference.id.clone());
                }
            }
        }
        for id in &requested_cast {
            let r = base
                .creative
                .references
                .iter()
                .find(|r| &r.id == id)
                .ok_or("cast_reference_missing")?;
            if !r.enabled {
                continue;
            }
            cast.push(json!({"story_names":[r.name],"reference_tag":format!("@{}",r.alias.trim_start_matches('@')),"description":r.description,"required":true,"default_language":r.voice.as_ref().map(|v|v.language.as_str()).unwrap_or(""),"voice_guidance":r.voice.as_ref().map(|v|v.description.as_str()).unwrap_or("")}));
            if request.operation != Operation::Outline {
                for image in r.images.iter().filter(|i| i.enabled) {
                    if !selections.iter().any(|i| i.asset == image.asset) {
                        selections.push(StoryImage {
                            pair_with: None,
                            id: image.id.clone(),
                            asset: image.asset.clone(),
                            label: r.name.clone(),
                            usage: ImageUse::SettingReference,
                        });
                    }
                }
            }
        }
        if request.operation != Operation::Outline {
            let scene = base
                .scene(request.scene_id.as_ref().ok_or("scene_not_found")?)
                .map_err(|_| "scene_not_found")?;
            for binding in scene.bindings.iter().filter(|b| b.enabled) {
                if binding.role == InputRole::VideoReference {
                    return Err("authoring_video_context_unavailable");
                }
                if base
                    .assets
                    .get(&binding.asset)
                    .is_some_and(|a| a.kind == MediaKind::Image)
                    && !selections.iter().any(|i| i.asset == binding.asset)
                {
                    selections.push(StoryImage {
                        pair_with: None,
                        id: binding.asset.0.clone(),
                        asset: binding.asset.clone(),
                        label: format!("{:?}", binding.role),
                        usage: ImageUse::SettingReference,
                    });
                }
            }
            for section in scene.sections.iter().filter(|s| {
                request.operation != Operation::EnhanceSection
                    || Some(&s.id) == request.section_id.as_ref()
            }) {
                for binding in base
                    .creative
                    .sections
                    .get(&section.id)
                    .into_iter()
                    .flat_map(|s| &s.images)
                {
                    if !selections.iter().any(|i| i.asset == binding.asset) {
                        selections.push(StoryImage {
                            pair_with: None,
                            id: binding.image_id.clone(),
                            asset: binding.asset.clone(),
                            label: "Section source image".into(),
                            usage: ImageUse::SettingReference,
                        });
                    }
                }
            }
        }
        let mut images = vec![];
        let mut ids = std::collections::BTreeSet::new();
        for selection in selections {
            if selection.id.is_empty() || !ids.insert(selection.id.clone()) {
                return Err("invalid_story_image");
            }
            if base
                .assets
                .get(&selection.asset)
                .is_none_or(|a| a.kind != MediaKind::Image)
            {
                return Err("story_image_missing");
            }
            let resource=session.document().resources.iter().find(|r|matches!(&r.kind,ResourceKind::Reference{asset_id,..} if asset_id==&selection.asset)).ok_or("reference_resource_missing")?;
            let mut file = session
                .resource_file(&resource.id)
                .map_err(|_| "resource_missing")?;
            crate::video::media::detect(&file)?;
            file.rewind().map_err(|_| "media_io_failed")?;
            let mut bytes = vec![];
            file.read_to_end(&mut bytes)
                .map_err(|_| "media_io_failed")?;
            let normalized = super::vision::normalize(&bytes)?;
            images.push((
                selection,
                ProviderImage {
                    mime: "image/jpeg".into(),
                    base64: STANDARD.encode(normalized.jpeg),
                },
            ));
        }
        Ok(Self {
            images,
            cast: json!(cast),
        })
    }
    pub fn selected(&self, ids: &[String]) -> Vec<ProviderImage> {
        self.images
            .iter()
            .filter(|(s, _)| ids.contains(&s.id))
            .map(|(_, p)| p.clone())
            .collect()
    }
}
