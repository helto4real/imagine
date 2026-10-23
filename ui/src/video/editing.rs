//! Pure Timeline gestures and authoring-only undo. Completed Takes are never rolled back.
use imagine_video_contract::{contract::*, editing::Edit};
use std::collections::BTreeSet;
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum Snap {
    None,
    Seconds,
    #[default]
    Frames,
}
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    #[default]
    TrimNeighbor,
    Ripple,
}
pub fn seconds(t: Time) -> f64 {
    t.num as f64 / t.den as f64
}
pub fn time(seconds: f64) -> Time {
    Time {
        num: (seconds.max(0.) * 1_000_000.).round() as u64,
        den: 1_000_000,
    }
    .normalized()
}
pub fn snapped(value: f64, snap: Snap, fps: Time) -> Time {
    match snap {
        Snap::None => time(value),
        Snap::Seconds => Time {
            num: value.max(0.).round() as u64,
            den: 1,
        },
        Snap::Frames => Time {
            num: ((value.max(0.) * seconds(fps)).round() as u64).saturating_mul(fps.den),
            den: fps.num,
        }
        .normalized(),
    }
}
pub fn move_scenes(
    project: &VideoProject,
    selected: &BTreeSet<SceneId>,
    delta: f64,
    snap: Snap,
) -> Vec<Edit> {
    let mut low = f64::NEG_INFINITY;
    let mut high = f64::INFINITY;
    for scene in project.scenes.iter().filter(|s| selected.contains(&s.id)) {
        let start = seconds(scene.start);
        let end = start + seconds(scene.duration);
        low = low.max(-start);
        for other in project.scenes.iter().filter(|s| !selected.contains(&s.id)) {
            let a = seconds(other.start);
            let b = a + seconds(other.duration);
            if b <= start + 1e-6 {
                low = low.max(b - start);
            } else if a >= end - 1e-6 {
                high = high.min(a - end);
            }
        }
    }
    if low > high || !low.is_finite() {
        return vec![];
    }
    let anchor = project
        .scenes
        .iter()
        .filter(|s| selected.contains(&s.id))
        .map(|s| seconds(s.start))
        .fold(f64::INFINITY, f64::min);
    let delta =
        (seconds(snapped(anchor + delta, snap, project.delivery.fps)) - anchor).clamp(low, high);
    project
        .scenes
        .iter()
        .filter(|s| selected.contains(&s.id))
        .map(|s| Edit::MoveScene {
            scene_id: s.id.clone(),
            start: time(seconds(s.start) + delta),
        })
        .collect()
}
pub fn section_gesture(
    scene: &Scene,
    id: &SectionId,
    edge: Option<bool>,
    value: f64,
    snap: Snap,
    fps: Time,
    mode: Mode,
) -> Vec<Edit> {
    let mut list = scene.sections.clone();
    list.sort_by(|a, b| seconds(a.start).total_cmp(&seconds(b.start)));
    let Some(index) = list.iter().position(|s| &s.id == id) else {
        return vec![];
    };
    let minimum = 0.25;
    let position = seconds(snapped(value, snap, fps));
    let limit = seconds(scene.duration);
    let start = seconds(list[index].start);
    let end = start + seconds(list[index].duration);
    let previous = index.checked_sub(1);
    let next = (index + 1 < list.len()).then_some(index + 1);
    let prev_end = previous
        .map(|i| seconds(list[i].start) + seconds(list[i].duration))
        .unwrap_or(0.);
    let next_start = next.map(|i| seconds(list[i].start)).unwrap_or(limit);
    match edge {
        None => {
            let last = (next_start - (end - start)).max(prev_end);
            list[index].start = time(position.clamp(prev_end, last));
        }
        Some(false) => {
            let touches = (start - prev_end).abs() < 1e-6;
            let lower = if mode == Mode::TrimNeighbor && (touches || position < prev_end) {
                previous
                    .map(|i| seconds(list[i].start) + minimum)
                    .unwrap_or(0.)
            } else {
                prev_end
            };
            let value = position.clamp(lower.min(end - minimum), end - minimum);
            list[index].start = time(value);
            list[index].duration = time(end - value);
            if mode == Mode::TrimNeighbor
                && let Some(i) = previous
                && (touches || position < prev_end)
            {
                list[i].duration = time(value - seconds(list[i].start));
            }
        }
        Some(true) if mode == Mode::Ripple => {
            let last = list
                .last()
                .map(|s| seconds(s.start) + seconds(s.duration))
                .unwrap_or(end);
            let delta =
                (position - end).clamp(-(end - start - minimum).max(0.), (limit - last).max(0.));
            list[index].duration = time(end + delta - start);
            for section in list.iter_mut().skip(index + 1) {
                section.start = time(seconds(section.start) + delta);
            }
        }
        Some(true) => {
            let touches = (end - next_start).abs() < 1e-6;
            let upper = if touches || position > next_start {
                next.map(|i| seconds(list[i].start) + seconds(list[i].duration) - minimum)
                    .unwrap_or(limit)
            } else {
                next_start
            };
            let value = position.clamp((start + minimum).min(upper), upper);
            list[index].duration = time(value - start);
            if let Some(i) = next
                && (touches || position > next_start)
            {
                let last = seconds(list[i].start) + seconds(list[i].duration);
                list[i].start = time(value);
                list[i].duration = time(last - value);
            }
        }
    }
    list.into_iter()
        .filter(|s| scene.sections.iter().find(|v| v.id == s.id) != Some(s))
        .map(|section| Edit::UpdateSection {
            scene_id: scene.id.clone(),
            section,
        })
        .collect()
}
pub fn difference(current: &VideoProject, desired: &VideoProject) -> Vec<Edit> {
    let mut commands = vec![];
    if current.default_model != desired.default_model {
        commands.push(Edit::SetDefaultModel {
            model: desired.default_model.clone(),
        });
    }
    if current.delivery != desired.delivery {
        commands.push(Edit::SetDelivery {
            delivery: desired.delivery.clone(),
        });
    }
    if current.global_prompt != desired.global_prompt {
        commands.push(Edit::SetGlobalPrompt {
            prompt: desired.global_prompt.clone(),
        });
    }
    if current.audio != desired.audio {
        commands.push(Edit::SetAudioSettings {
            settings: desired.audio.clone(),
        });
    }
    if current.audio_tracks != desired.audio_tracks {
        commands.push(Edit::SetAudioTracks {
            tracks: desired.audio_tracks.clone(),
        });
    }
    for scene in &current.scenes {
        if desired.scene(&scene.id).is_err() {
            commands.push(Edit::RemoveScene {
                scene_id: scene.id.clone(),
            });
        }
    }
    for desired in &desired.scenes {
        let Ok(current) = current.scene(&desired.id) else {
            let mut scene = desired.clone();
            scene.accepted_take = None;
            commands.push(Edit::AddScene {
                scene: Box::new(scene),
            });
            continue;
        };
        let id = desired.id.clone();
        if current.start != desired.start {
            commands.push(Edit::MoveScene {
                scene_id: id.clone(),
                start: desired.start,
            });
        }
        if current.duration != desired.duration {
            commands.push(Edit::SetSceneDuration {
                scene_id: id.clone(),
                duration: desired.duration,
            });
        }
        if current.description != desired.description {
            commands.push(Edit::SetSceneDescription {
                scene_id: id.clone(),
                description: desired.description.clone(),
            });
        }
        if current.model != desired.model {
            commands.push(Edit::SetSceneModel {
                scene_id: id.clone(),
                model: desired.model.clone(),
            });
        }
        for settings in &desired.settings {
            if !current.settings.contains(settings) {
                commands.push(Edit::SetModelSettings {
                    scene_id: id.clone(),
                    settings: settings.clone(),
                });
            }
        }
        if current.bindings != desired.bindings {
            commands.push(Edit::SetBindings {
                scene_id: id.clone(),
                bindings: desired.bindings.clone(),
            });
        }
        if current.continuity != desired.continuity {
            commands.push(Edit::SetContinuity {
                scene_id: id.clone(),
                continuity: desired.continuity.clone(),
            });
        }
        for section in &current.sections {
            if !desired.sections.iter().any(|s| s.id == section.id) {
                commands.push(Edit::RemoveSection {
                    scene_id: id.clone(),
                    section_id: section.id.clone(),
                });
            }
        }
        for section in &desired.sections {
            match current.sections.iter().find(|s| s.id == section.id) {
                None => commands.push(Edit::AddSection {
                    scene_id: id.clone(),
                    section: section.clone(),
                }),
                Some(old) if old != section => commands.push(Edit::UpdateSection {
                    scene_id: id.clone(),
                    section: section.clone(),
                }),
                _ => {}
            }
        }
    }
    let order = desired
        .scenes
        .iter()
        .map(|s| s.id.clone())
        .collect::<Vec<_>>();
    if current
        .scenes
        .iter()
        .map(|s| s.id.clone())
        .collect::<Vec<_>>()
        != order
    {
        commands.push(Edit::SetSceneOrder { scene_ids: order });
    }
    if current.assembly != desired.assembly {
        commands.push(Edit::SetAssembly {
            intent: desired.assembly.clone(),
        });
    }
    if current.creative != desired.creative {
        commands.push(Edit::SetCreative {
            creative: desired.creative.clone(),
        });
    }
    commands
}
#[derive(Default)]
pub struct History {
    pub undo: Vec<(VideoProject, VideoProject)>,
    pub redo: Vec<(VideoProject, VideoProject)>,
}
impl History {
    pub fn record(&mut self, before: VideoProject, after: VideoProject) {
        if difference(&before, &after).is_empty() {
            return;
        }
        self.undo.push((before, after));
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }
    pub fn undo_commands(&self, current: &VideoProject) -> Vec<Edit> {
        self.undo
            .last()
            .map(|(before, _)| difference(current, before))
            .unwrap_or_default()
    }
    pub fn redo_commands(&self, current: &VideoProject) -> Vec<Edit> {
        self.redo
            .last()
            .map(|(_, after)| difference(current, after))
            .unwrap_or_default()
    }
}
pub fn resize_scene(project: &VideoProject, scene: &Scene, duration: f64, mode: Mode) -> Vec<Edit> {
    let minimum = scene
        .sections
        .iter()
        .map(|s| seconds(s.start) + 0.25)
        .fold(0.25, f64::max);
    let duration = duration.max(minimum);
    let mut commands = vec![Edit::SetSceneDuration {
        scene_id: scene.id.clone(),
        duration: time(duration),
    }];
    for section in &scene.sections {
        if seconds(section.start) + seconds(section.duration) > duration {
            let mut section = section.clone();
            section.duration = time(duration - seconds(section.start));
            commands.push(Edit::UpdateSection {
                scene_id: scene.id.clone(),
                section,
            });
        }
    }
    if mode == Mode::Ripple {
        let delta = duration - seconds(scene.duration);
        for other in project.scenes.iter().filter(|s| {
            seconds(s.start) >= seconds(scene.start) + seconds(scene.duration) - 1e-6
                && s.id != scene.id
        }) {
            commands.push(Edit::MoveScene {
                scene_id: other.id.clone(),
                start: time(seconds(other.start) + delta),
            });
        }
    }
    commands
}

#[cfg(test)]
mod tests {
    use super::*;
    use imagine_video_contract::editing::{self as domain, EditRequest};
    fn project() -> VideoProject {
        let model = ModelSelection {
            adapter: "synthetic".into(),
            model: "test".into(),
            variant: "test".into(),
            profile: "test".into(),
            profile_revision: "1".into(),
        };
        VideoProject::create(CreateProject {
            schema_version: 1,
            default_model: model,
            delivery: DeliverySettings {
                width: 64,
                height: 64,
                fps: Time { num: 24, den: 1 },
            },
        })
        .unwrap()
    }
    fn scene(id: &str, start: f64, duration: f64) -> Scene {
        Scene {
            id: SceneId(id.into()),
            start: time(start),
            duration: time(duration),
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        }
    }
    fn apply(project: &mut VideoProject, commands: Vec<Edit>) {
        domain::apply(
            project,
            EditRequest {
                schema_version: 1,
                expected_project_revision: project.project_revision,
                commands,
            },
        )
        .unwrap();
    }
    #[test]
    fn multi_move_snaps_one_anchor_preserves_offsets_and_clamps_after_snap() {
        let mut p = project();
        p.scenes = vec![
            scene("a", 0.25, 1.),
            scene("b", 2., 1.),
            scene("fixed", 4.9, 1.),
        ];
        let selected = BTreeSet::from([SceneId("a".into()), SceneId("b".into())]);
        let commands = move_scenes(&p, &selected, 2.2, Snap::Seconds);
        apply(&mut p, commands);
        assert!((seconds(p.scenes[1].start) - seconds(p.scenes[0].start) - 1.75).abs() < 1e-6);
        assert!(seconds(p.scenes[1].start) + 1. <= 4.9 + 1e-6);
        assert_eq!(
            snapped(1. / 24., Snap::Frames, p.delivery.fps),
            Time { num: 1, den: 24 }
        );
    }
    #[test]
    fn neighbor_trim_and_ripple_preserve_director_section_semantics() {
        let mut p = project();
        let mut s = scene("a", 0., 4.);
        s.sections = vec![
            Section {
                id: SectionId("first".into()),
                start: time(0.),
                duration: time(1.),
                prompt: String::new(),
            },
            Section {
                id: SectionId("next".into()),
                start: time(1.),
                duration: time(1.),
                prompt: String::new(),
            },
        ];
        p.scenes.push(s.clone());
        apply(
            &mut p,
            section_gesture(
                &s,
                &SectionId("first".into()),
                Some(true),
                1.5,
                Snap::None,
                Time { num: 24, den: 1 },
                Mode::TrimNeighbor,
            ),
        );
        assert_eq!(p.scenes[0].sections[0].duration, time(1.5));
        assert_eq!(p.scenes[0].sections[1].start, time(1.5));
        assert_eq!(p.scenes[0].sections[1].duration, time(0.5));
        let mut p = project();
        p.scenes.push(s.clone());
        apply(
            &mut p,
            section_gesture(
                &s,
                &SectionId("first".into()),
                Some(true),
                1.5,
                Snap::None,
                Time { num: 24, den: 1 },
                Mode::Ripple,
            ),
        );
        assert_eq!(p.scenes[0].sections[1].start, time(1.5));
        assert_eq!(p.scenes[0].sections[1].duration, time(1.));
    }
    #[test]
    fn undo_restores_middle_scene_order_with_new_revision_without_rolling_back_take() {
        let mut p = project();
        p.scenes = vec![scene("a", 0., 2.), scene("b", 2., 2.), scene("c", 4., 2.)];
        let before = p.clone();
        apply(
            &mut p,
            vec![Edit::RemoveScene {
                scene_id: SceneId("b".into()),
            }],
        );
        let mut history = History::default();
        history.record(before.clone(), p.clone());
        let take = AcceptedTake {
            id: TakeId("new-result".into()),
            model: p.default_model.clone(),
            source_fingerprint: "new".into(),
        };
        p.scenes[0].accepted_take = Some(take.clone());
        let commands = history.undo_commands(&p);
        apply(&mut p, commands);
        assert_eq!(
            p.scenes.iter().map(|s| &s.id.0).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(p.project_revision, 2);
        assert_eq!(p.scenes[0].accepted_take, Some(take));
    }
}
