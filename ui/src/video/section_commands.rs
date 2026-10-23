//! Section operations preserve authored bindings. No generated media is removed.
use super::editing::{Snap, seconds, time};
use super::*;

pub(super) fn split_after(
    project: &VideoProject,
    mut pending: Vec<Edit>,
    id: &SectionId,
    snap: Snap,
) -> Result<Vec<Edit>, Issue> {
    let mut current = project.clone();
    if !pending.is_empty() {
        imagine_video_contract::editing::apply(
            &mut current,
            EditRequest {
                schema_version: 1,
                expected_project_revision: project.project_revision,
                commands: pending.clone(),
            },
        )?;
    }
    pending.extend(split(&current, id, snap));
    Ok(pending)
}

pub(super) fn split(project: &VideoProject, id: &SectionId, snap: Snap) -> Vec<Edit> {
    let Some(scene) = project
        .scenes
        .iter()
        .find(|s| s.sections.iter().any(|s| &s.id == id))
    else {
        return vec![];
    };
    let section = scene.sections.iter().find(|s| &s.id == id).unwrap();
    let start = seconds(section.start);
    let end = start + seconds(section.duration);
    let middle = seconds(editing::snapped(
        (start + end) / 2.,
        snap,
        project.delivery.fps,
    ));
    if middle <= start || middle >= end {
        return vec![];
    }
    let mut left = section.clone();
    left.duration = time(middle - start);
    let mut right = section.clone();
    right.id = SectionId(format!("section_{}", uuid::Uuid::new_v4().simple()));
    right.start = time(middle);
    right.duration = time(end - middle);
    let mut commands = vec![
        Edit::UpdateSection {
            scene_id: scene.id.clone(),
            section: left,
        },
        Edit::AddSection {
            scene_id: scene.id.clone(),
            section: right.clone(),
        },
    ];
    if let Some(source) = project.creative.sections.get(id) {
        let mut creative = project.creative.clone();
        let mut left = source.clone();
        let mut right_source = source.clone();
        let ratio = (middle - start) / (end - start);
        for (a, b) in left.videos.iter_mut().zip(&mut right_source.videos) {
            if let Some(video) = &mut a.video
                && let Some(out) = video.source_out
            {
                let input = video.source_in.map(seconds).unwrap_or(0.);
                let at = time(input + (seconds(out) - input) * ratio);
                video.source_out = Some(at);
                if let Some(video) = &mut b.video {
                    video.source_in = Some(at);
                }
            }
        }
        creative.sections.insert(id.clone(), left);
        creative.sections.insert(right.id, right_source);
        commands.push(Edit::SetCreative { creative });
    }
    commands
}
impl Timeline {
    pub(super) fn delete_selection(&mut self) -> Vec<Edit> {
        let Some(project) = self.document.as_ref().map(|d| &d.project) else {
            return vec![];
        };
        let items = self.selection_items(project);
        let scenes = items
            .iter()
            .filter_map(|i| match i {
                selection::Item::Scene(id) => Some(id.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let mut edits = scenes
            .iter()
            .cloned()
            .map(|scene_id| Edit::RemoveScene { scene_id })
            .collect::<Vec<_>>();
        if !self.audio_selected.is_empty() {
            edits.extend(
                project
                    .audio_tracks
                    .iter()
                    .flat_map(|t| &t.clips)
                    .filter(|c| self.audio_selected.contains(&c.id))
                    .map(|c| Edit::RemoveAudioClip {
                        clip_id: c.id.clone(),
                    })
                    .collect::<Vec<_>>(),
            );
        }
        if !self.section_selection.is_empty() {
            for s in &project.scenes {
                if scenes.contains(&s.id) {
                    continue;
                }
                edits.extend(
                    s.sections
                        .iter()
                        .filter(|section| self.section_selection.contains(&section.id))
                        .map(|section| Edit::RemoveSection {
                            scene_id: s.id.clone(),
                            section_id: section.id.clone(),
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }
        if !edits.is_empty() {
            self.after_edit = Some(PostAction::ClearSelection);
        }
        edits
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_retains_text_images_dialogue_and_interpolates_only_trimmed_video_ranges() {
        let mut p = super::super::media_picker::tests::project();
        let id = p.scenes[0].sections[0].id.clone();
        p.scenes[0].sections[0].start = time(0.);
        p.scenes[0].sections[0].duration = time(1.);
        p.creative.sections.insert(
            id.clone(),
            imagine_video_contract::creative::SectionSource {
                videos: vec![Binding {
                    id: BindingId("own-video".into()),
                    asset: AssetId("video".into()),
                    role: InputRole::VisualReference,
                    enabled: true,
                    video: Some(VideoReferenceSettings {
                        source_in: Some(time(2.)),
                        source_out: Some(time(6.)),
                        ..Default::default()
                    }),
                }],
                ..Default::default()
            },
        );
        let commands = split(&p, &id, Snap::None);
        assert_eq!(commands.len(), 3);
        let Edit::SetCreative { creative } = commands.last().unwrap() else {
            panic!()
        };
        let left = &creative.sections[&id];
        let right = creative
            .sections
            .iter()
            .find(|(key, _)| *key != &id)
            .unwrap()
            .1;
        assert_eq!(
            left.videos[0].video.as_ref().unwrap().source_out,
            Some(time(4.))
        );
        assert_eq!(
            right.videos[0].video.as_ref().unwrap().source_in,
            Some(time(4.))
        );
        let Edit::AddSection { section, .. } = &commands[1] else {
            panic!()
        };
        assert_eq!(section.start, time(0.5));
        assert_eq!(section.duration, time(0.5));
        assert_eq!(section.prompt, p.scenes[0].sections[0].prompt);
        assert!(split(&p, &SectionId("missing".into()), Snap::Frames).is_empty());
        let mut pending = p.creative.clone();
        pending.references[0].description = "Own pending edit".into();
        let commands = split_after(
            &p,
            vec![Edit::SetCreative { creative: pending }],
            &id,
            Snap::None,
        )
        .unwrap();
        let revision = p.project_revision;
        imagine_video_contract::editing::apply(
            &mut p,
            EditRequest {
                schema_version: 1,
                expected_project_revision: revision,
                commands,
            },
        )
        .unwrap();
        assert_eq!(p.creative.references[0].description, "Own pending edit");
        assert_eq!(p.scenes[0].sections.len(), 2);
    }
}
