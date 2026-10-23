//! Global Director Section operations projected into native Scene-local Sections.
use super::editing::{Snap, seconds, time};
use super::*;
use std::collections::BTreeMap;
fn spans(p: &VideoProject) -> Vec<(SectionId, f64, f64)> {
    let mut spans = p
        .scenes
        .iter()
        .flat_map(|s| {
            s.sections.iter().map(move |v| {
                (
                    v.id.clone(),
                    seconds(s.start) + seconds(v.start),
                    seconds(s.start) + seconds(v.start) + seconds(v.duration),
                )
            })
        })
        .collect::<Vec<_>>();
    spans.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.total_cmp(&b.2)));
    spans
}
pub(super) fn duration(p: &VideoProject) -> f64 {
    p.creative
        .timeline_duration
        .map(seconds)
        .unwrap_or_else(|| super::navigation::duration(p))
}
// Rebase Scene-local sections without losing their global position. Resizing
// preserves authored Scene space; movement/distribution retains its existing
// envelope semantics. Neighbour section envelopes still bound shared edges.
fn retime(
    p: &mut VideoProject,
    updates: &BTreeMap<SectionId, (f64, f64)>,
    preserve_scene_space: bool,
) {
    let envelopes = p
        .scenes
        .iter()
        .map(|scene| {
            let start = seconds(scene.start);
            let section_bounds = scene
                .sections
                .iter()
                .map(|section| {
                    updates.get(&section.id).copied().unwrap_or((
                        start + seconds(section.start),
                        start + seconds(section.start) + seconds(section.duration),
                    ))
                })
                .collect::<Vec<_>>();
            (
                scene.id.clone(),
                start,
                section_bounds
                    .iter()
                    .map(|v| v.0)
                    .reduce(f64::min)
                    .unwrap_or(start),
                section_bounds
                    .iter()
                    .map(|v| v.1)
                    .reduce(f64::max)
                    .unwrap_or(start + seconds(scene.duration)),
            )
        })
        .collect::<Vec<_>>();
    for scene in &mut p.scenes {
        if !scene.sections.iter().any(|s| updates.contains_key(&s.id)) {
            continue;
        }
        let global = scene
            .sections
            .iter()
            .map(|s| {
                updates.get(&s.id).copied().unwrap_or((
                    seconds(scene.start) + seconds(s.start),
                    seconds(scene.start) + seconds(s.start) + seconds(s.duration),
                ))
            })
            .collect::<Vec<_>>();
        let mut start = global.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
        let mut end = global.iter().map(|v| v.1).fold(0., f64::max);
        if preserve_scene_space {
            let old_start = seconds(scene.start);
            let old_end = old_start + seconds(scene.duration);
            let previous_end = envelopes
                .iter()
                .filter(|(id, at, _, _)| id != &scene.id && *at < old_start)
                .map(|(_, _, _, end)| *end)
                .fold(0., f64::max);
            let next_start = envelopes
                .iter()
                .filter(|(id, at, _, _)| id != &scene.id && *at > old_start)
                .map(|(_, _, start, _)| *start)
                .fold(f64::INFINITY, f64::min);
            start = start.min(old_start).max(previous_end);
            end = end.max(old_end).min(next_start);
        }
        scene.start = time(start);
        scene.duration = time(end - start);
        for (s, (a, b)) in scene.sections.iter_mut().zip(global) {
            s.start = time(a - start);
            s.duration = time(b - a);
        }
    }
    p.scenes
        .sort_by(|a, b| seconds(a.start).total_cmp(&seconds(b.start)));
}
// Director trim/ripple works on the global Section track, across Shot owners.
pub(super) fn resize(
    p: &VideoProject,
    id: &SectionId,
    end_edge: bool,
    value: f64,
    snap: Snap,
    mode: editing::Mode,
) -> Vec<Edit> {
    let mut all = spans(p);
    let Some(i) = all.iter().position(|s| &s.0 == id) else {
        return vec![];
    };
    let position = seconds(editing::snapped(value, snap, p.delivery.fps));
    let (start, end) = (all[i].1, all[i].2);
    let previous = i.checked_sub(1);
    let next = (i + 1 < all.len()).then_some(i + 1);
    let minimum = 0.25;
    let limit = duration(p);
    let clamp = |v: f64, low: f64, high: f64| v.min(high).max(low);
    if !end_edge {
        let previous_end = previous.map(|n| all[n].2).unwrap_or(0.);
        if mode == editing::Mode::TrimNeighbor
            && let Some(n) = previous
            && ((start - previous_end).abs() < 1e-6 || position < previous_end)
        {
            let at = clamp(position, all[n].1 + minimum, end - minimum);
            all[n].2 = at;
            all[i].1 = at;
        } else {
            all[i].1 = clamp(position, previous_end, end - minimum);
        }
    } else if mode == editing::Mode::Ripple {
        let last = all.last().unwrap().2;
        let requested = position.max(start + minimum);
        let delta = clamp(
            requested - end,
            -(end - start - minimum).max(0.),
            limit - last,
        );
        all[i].2 += delta;
        for section in all.iter_mut().skip(i + 1) {
            section.1 += delta;
            section.2 += delta;
        }
    } else if let Some(n) = next
        && ((end - all[n].1).abs() < 1e-6 || position > all[n].1)
    {
        let at = clamp(position, start + minimum, all[n].2 - minimum);
        all[i].2 = at;
        all[n].1 = at;
    } else {
        all[i].2 = clamp(
            position,
            start + minimum,
            next.map(|n| all[n].1).unwrap_or(limit),
        );
    }
    let updates = all.into_iter().map(|(id, a, b)| (id, (a, b))).collect();
    let mut desired = p.clone();
    retime(&mut desired, &updates, true);
    editing::difference(p, &desired)
}
pub(super) fn move_items(
    p: &VideoProject,
    sections: &[SectionId],
    audio: &BTreeSet<AudioClipId>,
    dragged: f64,
    delta: f64,
    snap: Snap,
) -> Vec<Edit> {
    move_mixed(p, sections, audio, &BTreeSet::new(), dragged, delta, snap)
}
fn move_mixed(
    p: &VideoProject,
    sections: &[SectionId],
    audio: &BTreeSet<AudioClipId>,
    scenes: &BTreeSet<SceneId>,
    dragged: f64,
    delta: f64,
    snap: Snap,
) -> Vec<Edit> {
    let scenes = scenes
        .iter()
        .filter(|id| p.scene(id).is_ok_and(|s| s.sections.is_empty()))
        .cloned()
        .collect::<BTreeSet<_>>();
    let all = spans(p);
    let limit = duration(p);
    let mut low = f64::NEG_INFINITY;
    let mut high = f64::INFINITY;
    let mut movable = false;
    for (id, a, b) in &all {
        if !sections.contains(id) {
            continue;
        }
        movable = true;
        low = low.max(-a);
        high = high.min(limit - b);
        for (other, c, d) in &all {
            if sections.contains(other) {
                continue;
            }
            if d <= a {
                low = low.max(d - a)
            } else if c >= b {
                high = high.min(c - b)
            }
        }
    }
    for c in p
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .filter(|c| audio.contains(&c.id) && !c.locked)
    {
        movable = true;
        low = low.max(-seconds(c.start));
        high = high.min(limit - seconds(c.start) - seconds(c.duration));
    }
    for scene in p
        .scenes
        .iter()
        .filter(|s| scenes.contains(&s.id) && s.sections.is_empty())
    {
        let a = seconds(scene.start);
        let b = a + seconds(scene.duration);
        movable = true;
        low = low.max(-a);
        high = high.min(limit - b);
        for other in p.scenes.iter().filter(|s| !scenes.contains(&s.id)) {
            let c = seconds(other.start);
            let d = c + seconds(other.duration);
            if d <= a {
                low = low.max(d - a)
            } else if c >= b {
                high = high.min(c - b)
            }
        }
    }
    if !movable || low > high {
        return vec![];
    }
    let delta = (seconds(editing::snapped(dragged + delta, snap, p.delivery.fps)) - dragged)
        .clamp(low, high);
    let updates = all
        .into_iter()
        .filter(|(id, _, _)| sections.contains(id))
        .map(|(id, a, b)| (id, (a + delta, b + delta)))
        .collect();
    let mut desired = p.clone();
    for edit in editing::move_scenes(p, &scenes, delta, Snap::None) {
        if let Edit::MoveScene { scene_id, start } = edit
            && let Some(s) = desired.scenes.iter_mut().find(|s| s.id == scene_id)
        {
            s.start = start;
        }
    }
    retime(&mut desired, &updates, false);
    for c in desired
        .audio_tracks
        .iter_mut()
        .flat_map(|t| &mut t.clips)
        .filter(|c| audio.contains(&c.id) && !c.locked)
    {
        c.start = time(seconds(c.start) + delta);
    }
    auto_lanes(&mut desired);
    editing::difference(p, &desired)
}
fn auto_lanes(p: &mut VideoProject) {
    for t in &mut p.audio_tracks {
        t.clips
            .sort_by(|a, b| seconds(a.start).total_cmp(&seconds(b.start)));
        let mut ends = vec![];
        for c in &mut t.clips {
            let lane = ends
                .iter()
                .position(|v| *v <= seconds(c.start))
                .unwrap_or(ends.len());
            if lane == ends.len() {
                ends.push(0.)
            }
            ends[lane] = seconds(c.start) + seconds(c.duration);
            c.lane = lane as u32;
        }
        t.clips.sort_by(|a, b| {
            a.lane
                .cmp(&b.lane)
                .then_with(|| seconds(a.start).total_cmp(&seconds(b.start)))
        });
    }
}
pub(super) struct Duplicate {
    pub edits: Vec<Edit>,
    pub sections: Vec<SectionId>,
    pub audio: Vec<AudioClipId>,
}
pub(super) fn add(p: &VideoProject, selected: Option<&SceneId>, snap: Snap) -> Duplicate {
    let target = selected
        .and_then(|id| p.scene(id).ok())
        .filter(|s| p.scene_kind(&s.id) != imagine_video_contract::creative::SceneKind::Imported);
    let length = duration(p).min(1.);
    let preferred = target
        .map(|s| {
            s.sections
                .iter()
                .map(|v| seconds(s.start) + seconds(v.start) + seconds(v.duration))
                .reduce(f64::max)
                .unwrap_or(seconds(s.start))
        })
        .unwrap_or(0.);
    let preferred = seconds(editing::snapped(preferred, snap, p.delivery.fps));
    let mut gaps = vec![];
    let mut cursor = 0.;
    for (_, a, b) in spans(p) {
        if a > cursor {
            gaps.push((cursor, a));
        }
        cursor = cursor.max(b);
    }
    if cursor < duration(p) {
        gaps.push((cursor, duration(p)));
    }
    let at = gaps
        .iter()
        .find_map(|(a, b)| {
            let at = a.max(preferred);
            (*b - at >= length).then_some(at)
        })
        .or_else(|| {
            target
                .is_none()
                .then(|| {
                    gaps.iter()
                        .find_map(|(a, b)| (*b - *a >= length).then_some(*a))
                })
                .flatten()
        });
    let Some(at) = at else {
        return Duplicate {
            edits: vec![],
            sections: vec![],
            audio: vec![],
        };
    };
    let mut desired = p.clone();
    let id = SectionId(format!("section_{}", uuid::Uuid::new_v4().simple()));
    let section = Section {
        id: id.clone(),
        start: time(0.),
        duration: time(length),
        prompt: String::new(),
    };
    if let Some(owner) = target {
        let owner = desired
            .scenes
            .iter_mut()
            .find(|s| s.id == owner.id)
            .unwrap();
        let new_start = owner
            .sections
            .iter()
            .map(|s| seconds(owner.start) + seconds(s.start))
            .fold(at, f64::min);
        let new_end = owner
            .sections
            .iter()
            .map(|s| seconds(owner.start) + seconds(s.start) + seconds(s.duration))
            .fold(at + length, f64::max);
        for v in &mut owner.sections {
            v.start = time(seconds(owner.start) + seconds(v.start) - new_start);
        }
        owner.start = time(new_start);
        owner.duration = time(new_end - new_start);
        owner.sections.push(Section {
            start: time(at - new_start),
            ..section
        });
    } else {
        desired.scenes.push(Scene {
            id: SceneId(format!("scene_{}", uuid::Uuid::new_v4().simple())),
            start: time(at),
            duration: time(length),
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![section],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
    }
    desired
        .scenes
        .sort_by(|a, b| seconds(a.start).total_cmp(&seconds(b.start)));
    Duplicate {
        edits: editing::difference(p, &desired),
        sections: vec![id],
        audio: vec![],
    }
}
pub(super) fn duplicate(
    p: &VideoProject,
    sections: &[SectionId],
    audio: &BTreeSet<AudioClipId>,
) -> Duplicate {
    let empty = || Duplicate {
        edits: vec![],
        sections: vec![],
        audio: vec![],
    };
    let all = spans(p);
    let chosen = all
        .iter()
        .filter(|(id, _, _)| sections.contains(id))
        .collect::<Vec<_>>();
    let clips = p
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .filter(|c| audio.contains(&c.id))
        .collect::<Vec<_>>();
    if chosen.is_empty() && clips.is_empty() {
        return empty();
    }
    let limit = duration(p);
    let delta = if !chosen.is_empty() {
        let start = chosen.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
        let end = chosen.iter().map(|v| v.2).fold(0., f64::max);
        let width = end - start;
        let mut gaps = vec![];
        let mut cursor = 0.;
        for (_, a, b) in &all {
            if *a > cursor {
                gaps.push((cursor, *a));
            }
            cursor = cursor.max(*b);
        }
        if cursor < limit {
            gaps.push((cursor, limit));
        }
        let at = gaps
            .iter()
            .find_map(|(a, b)| {
                let at = a.max(end);
                (*b - at >= width).then_some(at)
            })
            .or_else(|| {
                gaps.iter()
                    .find_map(|(a, b)| (*b - *a >= width).then_some(*a))
            });
        let Some(at) = at else { return empty() };
        at - start
    } else {
        let a = clips
            .iter()
            .map(|c| seconds(c.start))
            .fold(f64::INFINITY, f64::min);
        let b = clips
            .iter()
            .map(|c| seconds(c.start) + seconds(c.duration))
            .fold(0., f64::max);
        b - a
    };
    if clips.iter().any(|c| {
        seconds(c.start) + delta < 0. || seconds(c.start) + seconds(c.duration) + delta > limit
    }) {
        return empty();
    }
    let mut desired = p.clone();
    let mut new_sections = vec![];
    let mut new_audio = vec![];
    for (id, a, b) in chosen {
        let source = p
            .scenes
            .iter()
            .flat_map(|s| &s.sections)
            .find(|s| &s.id == id)
            .unwrap();
        let mut section = source.clone();
        section.id = SectionId(format!("section_{}", uuid::Uuid::new_v4().simple()));
        section.start = time(0.);
        if let Some(bindings) = p.creative.sections.get(id) {
            desired
                .creative
                .sections
                .insert(section.id.clone(), bindings.clone());
        }
        new_sections.push(section.id.clone());
        desired.scenes.push(Scene {
            id: SceneId(format!("scene_{}", uuid::Uuid::new_v4().simple())),
            start: time(a + delta),
            duration: time(b - a),
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![],
            sections: vec![section],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        });
    }
    if !clips.is_empty() {
        if desired.audio_tracks.is_empty() {
            desired.audio_tracks.push(AudioTrack {
                id: AudioTrackId("audio_track_001".into()),
                clips: vec![],
            });
        }
        for c in clips {
            let mut copy = c.clone();
            copy.id = AudioClipId(format!("audio_clip_{}", uuid::Uuid::new_v4().simple()));
            copy.start = time(seconds(copy.start) + delta);
            new_audio.push(copy.id.clone());
            desired.audio_tracks[0].clips.push(copy);
        }
        auto_lanes(&mut desired);
    }
    desired
        .scenes
        .sort_by(|a, b| seconds(a.start).total_cmp(&seconds(b.start)));
    Duplicate {
        edits: editing::difference(p, &desired),
        sections: new_sections,
        audio: new_audio,
    }
}
pub(super) fn prepared(p: &VideoProject, pending: Vec<Edit>) -> Result<VideoProject, Issue> {
    let mut current = p.clone();
    if !pending.is_empty() {
        imagine_video_contract::editing::apply(
            &mut current,
            EditRequest {
                schema_version: 1,
                expected_project_revision: p.project_revision,
                commands: pending,
            },
        )?;
    }
    Ok(current)
}
impl Timeline {
    pub(super) fn move_selection(&self, p: &VideoProject, dragged: f64, delta: f64) -> Vec<Edit> {
        let scenes: BTreeSet<SceneId> = self
            .selection_items(p)
            .iter()
            .filter_map(|item| match item {
                selection::Item::Scene(id) => Some(id.clone()),
                _ => None,
            })
            .collect();
        if scenes.is_empty() {
            move_items(
                p,
                &self.section_selection,
                &self.audio_selected,
                dragged,
                delta,
                self.snap,
            )
        } else {
            move_mixed(
                p,
                &self.section_selection,
                &self.audio_selected,
                &scenes,
                dragged,
                delta,
                self.snap,
            )
        }
    }
    pub(super) fn duplicate_selection(&mut self, p: &VideoProject) -> Vec<Edit> {
        match prepared(p, self.draft_edits()) {
            Ok(current) => {
                let result = duplicate(&current, &self.section_selection, &self.audio_selected);
                if result.edits.is_empty() {
                    return vec![];
                }
                self.after_edit = Some(PostAction::SelectGroup(result.sections, result.audio));
                let mut edits = editing::difference(p, &current);
                edits.extend(result.edits);
                edits
            }
            Err(e) => {
                self.error = Some(format!("Could not duplicate selection: {}", e.code));
                vec![]
            }
        }
    }
}
pub(super) fn overflow(p: &VideoProject) -> bool {
    spans(p).iter().any(|s| s.2 > duration(p))
}
pub(super) fn can_fit_last(p: &VideoProject) -> bool {
    spans(p)
        .last()
        .is_some_and(|s| s.2 > duration(p) && duration(p) > s.1)
}
pub(super) fn fit(p: &VideoProject, all: bool) -> Vec<Edit> {
    let spans = spans(p);
    let limit = duration(p);
    let maximum = spans.iter().map(|s| s.2).fold(0., f64::max);
    if !overflow(p) || (!all && !can_fit_last(p)) {
        return vec![];
    }
    let updates = if all {
        spans
            .into_iter()
            .map(|(id, a, b)| (id, (a * limit / maximum, b * limit / maximum)))
            .collect()
    } else {
        let (id, a, _) = spans.last().unwrap();
        BTreeMap::from([(id.clone(), (*a, limit))])
    };
    let mut desired = p.clone();
    retime(&mut desired, &updates, false);
    editing::difference(p, &desired)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn project() -> VideoProject {
        let mut p = super::super::media_picker::tests::project();
        let template = p.scenes[0].clone();
        p.scenes.clear();
        p.creative.sections.clear();
        p.creative.timeline_duration = Some(time(12.));
        for (i, a, b) in [(0, 0., 1.), (1, 3., 4.), (2, 10., 11.)] {
            let mut s = template.clone();
            s.id = SceneId(format!("own_scene_{i}"));
            s.start = time(a);
            s.duration = time(b - a);
            s.sections = vec![Section {
                id: SectionId(format!("own_section_{i}")),
                start: time(0.),
                duration: time(b - a),
                prompt: format!("Own {i}"),
            }];
            p.scenes.push(s);
        }
        p
    }
    fn apply(p: &mut VideoProject, commands: Vec<Edit>) {
        if commands.is_empty() {
            return;
        }
        imagine_video_contract::editing::apply(
            p,
            EditRequest {
                schema_version: 1,
                expected_project_revision: p.project_revision,
                commands,
            },
        )
        .unwrap();
    }
    #[test]
    fn resizing_section_preserves_ten_second_scene_and_authored_gaps() {
        for mode in [editing::Mode::TrimNeighbor, editing::Mode::Ripple] {
            for (start, length, end_edge, position) in [
                (0., 2., true, 4.),
                (0., 10., true, 3.),
                (2., 2., false, 3.),
                (2., 2., false, 1.),
            ] {
                let mut p = project();
                p.scenes.truncate(1);
                p.scenes[0].duration = time(10.);
                p.scenes[0].sections.truncate(1);
                p.scenes[0].sections[0].start = time(start);
                p.scenes[0].sections[0].duration = time(length);
                p.creative.timeline_duration = Some(time(10.));
                let before = p.clone();
                let id = p.scenes[0].sections[0].id.clone();
                let edits = resize(&p, &id, end_edge, position, Snap::None, mode);
                apply(&mut p, edits);
                assert_eq!(p.scenes[0].start, time(0.));
                assert_eq!(p.scenes[0].duration, time(10.));
                assert_ne!(p.scenes[0].sections, before.scenes[0].sections);
                let undo = editing::difference(&p, &before);
                apply(&mut p, undo);
                assert_eq!(p.scenes, before.scenes);
            }
        }
    }
    #[test]
    fn add_rebases_selected_owner_from_actual_sections_without_losing_metadata() {
        let mut p = project();
        p.scenes[0].duration = time(8.);
        p.scenes[0].sections[0].start = time(3.);
        p.scenes[0].description = "Keep own authored metadata".into();
        p.scenes.remove(1);
        let owner = p.scenes[0].id.clone();
        let result = add(&p, Some(&owner), Snap::None);
        apply(&mut p, result.edits);
        let s = p.scene(&owner).unwrap();
        assert_eq!((s.start, s.duration), (time(3.), time(2.)));
        assert_eq!(s.sections[0].start, time(0.));
        assert_eq!(s.sections[1].start, time(1.));
        assert_eq!(s.description, "Keep own authored metadata");
    }
    #[test]
    fn operations_match_the_executed_director_oracle() {
        let oracle: Value = serde_json::from_str(include_str!("group8-director.json")).unwrap();
        for row in oracle["cases"].as_array().unwrap() {
            let name = row["name"].as_str().unwrap();
            let mut p = project();
            if name.starts_with("resize_") || name.starts_with("add_") {
                let template = p.scenes[0].clone();
                p.scenes.clear();
                for (i, range) in row["input"].as_array().unwrap().iter().enumerate() {
                    let a = range[0].as_f64().unwrap();
                    let b = range[1].as_f64().unwrap();
                    let mut s = template.clone();
                    s.id = SceneId(format!("own_scene_{i}"));
                    s.start = time(a);
                    s.duration = time(b - a);
                    s.sections = vec![Section {
                        id: SectionId(format!("own_section_{i}")),
                        start: time(0.),
                        duration: time(b - a),
                        prompt: format!("Own {i}"),
                    }];
                    p.scenes.push(s);
                }
                if name.starts_with("resize_") {
                    let commands = resize(
                        &p,
                        &SectionId(row["id"].as_str().unwrap().into()),
                        row["edge"] == "end",
                        row["value"].as_f64().unwrap(),
                        Snap::None,
                        if row["mode"] == "Ripple Edit" {
                            editing::Mode::Ripple
                        } else {
                            editing::Mode::TrimNeighbor
                        },
                    );
                    apply(&mut p, commands);
                    for ((id, a, b), expected) in
                        spans(&p).iter().zip(row["sections"].as_array().unwrap())
                    {
                        assert_eq!(id.0, expected[0].as_str().unwrap());
                        assert!(
                            (a - expected[1].as_f64().unwrap()).abs() < 1e-6
                                && (b - expected[2].as_f64().unwrap()).abs() < 1e-6,
                            "{name}"
                        );
                    }
                } else {
                    let selected = row["owner"]
                        .as_u64()
                        .map(|i| SceneId(format!("own_scene_{i}")));
                    if row["imported"] == true {
                        p.creative.scene_kinds.insert(
                            selected.clone().unwrap(),
                            imagine_video_contract::creative::SceneKind::Imported,
                        );
                    }
                    let result = add(&p, selected.as_ref(), Snap::None);
                    let ids = result.sections;
                    apply(&mut p, result.edits);
                    let actual = spans(&p).into_iter().find(|v| ids.contains(&v.0));
                    if let Some(expected) = row["added"].as_array() {
                        let (_, a, b) = actual.unwrap();
                        assert_eq!(
                            (a, b),
                            (expected[0].as_f64().unwrap(), expected[1].as_f64().unwrap()),
                            "{name}"
                        );
                    } else {
                        assert!(actual.is_none(), "{name}");
                    }
                    if let Some(selected) = selected {
                        let s = p.scene(&selected).unwrap();
                        let expected = &row["owner_timing"];
                        assert_eq!(
                            (seconds(s.start), seconds(s.start) + seconds(s.duration)),
                            (
                                expected["start_time"].as_f64().unwrap(),
                                expected["end_time"].as_f64().unwrap()
                            ),
                            "{name}"
                        );
                    }
                }
            } else if name.starts_with("duplicate_") {
                let template = p.scenes[0].clone();
                p.scenes.clear();
                for (i, range) in row["input"].as_array().unwrap().iter().enumerate() {
                    let a = range[0].as_f64().unwrap();
                    let b = range[1].as_f64().unwrap();
                    let mut s = template.clone();
                    s.id = SceneId(format!("own_scene_{i}"));
                    s.start = time(a);
                    s.duration = time(b - a);
                    s.sections[0].id = SectionId(format!("own_section_{i}"));
                    s.sections[0].start = time(0.);
                    s.sections[0].duration = time(b - a);
                    p.scenes.push(s);
                }
                let selected = row["selected"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| SectionId(v.as_str().unwrap().into()))
                    .collect::<Vec<_>>();
                let result = duplicate(&p, &selected, &BTreeSet::new());
                let ids = result.sections.clone();
                apply(&mut p, result.edits);
                let actual = spans(&p)
                    .into_iter()
                    .filter(|v| ids.contains(&v.0))
                    .collect::<Vec<_>>();
                let expected = row["copied"].as_array().unwrap();
                assert_eq!(actual.len(), expected.len(), "{name}");
                for ((_, a, b), expected) in actual.iter().zip(expected) {
                    assert!(
                        (a - expected[0].as_f64().unwrap()).abs() < 1e-6
                            && (b - expected[1].as_f64().unwrap()).abs() < 1e-6,
                        "{name}"
                    );
                }
            } else if name == "mixed_move_upper_bound" {
                let audio = AudioClip::new(
                    AudioClipId("own_audio".into()),
                    time(1.),
                    time(1.),
                    &p.audio,
                );
                let mut locked = audio.clone();
                locked.id = AudioClipId("own_locked".into());
                locked.start = time(5.);
                locked.locked = true;
                p.audio_tracks = vec![AudioTrack {
                    id: AudioTrackId("own_track".into()),
                    clips: vec![audio, locked],
                }];
                let edits = move_items(
                    &p,
                    &[
                        SectionId("own_section_0".into()),
                        SectionId("own_section_1".into()),
                    ],
                    &BTreeSet::from([
                        AudioClipId("own_audio".into()),
                        AudioClipId("own_locked".into()),
                    ]),
                    0.,
                    100.,
                    Snap::None,
                );
                apply(&mut p, edits);
                for ((id, a, b), expected) in
                    spans(&p).iter().zip(row["sections"].as_array().unwrap())
                {
                    assert_eq!(id.0, expected[0].as_str().unwrap());
                    assert_eq!(
                        (*a, *b),
                        (expected[1].as_f64().unwrap(), expected[2].as_f64().unwrap())
                    );
                }
                for expected in row["audio"].as_array().unwrap() {
                    let c = p
                        .audio_tracks
                        .iter()
                        .flat_map(|t| &t.clips)
                        .find(|c| c.id.0 == expected[0].as_str().unwrap())
                        .unwrap();
                    assert_eq!(
                        (
                            seconds(c.start),
                            seconds(c.start) + seconds(c.duration),
                            c.lane
                        ),
                        (
                            expected[1].as_f64().unwrap(),
                            expected[2].as_f64().unwrap(),
                            expected[3].as_u64().unwrap() as u32
                        )
                    );
                }
            } else if name.starts_with("fit_") {
                p.creative.timeline_duration = Some(time(row["duration"].as_f64().unwrap()));
                assert_eq!(overflow(&p), row["overflow"].as_bool().unwrap());
                assert_eq!(can_fit_last(&p), row["can_last"].as_bool().unwrap());
                let edits = fit(&p, row["all"].as_bool().unwrap());
                assert_eq!(
                    !edits.is_empty(),
                    row["changed"].as_bool().unwrap(),
                    "{name}"
                );
                apply(&mut p, edits);
                for ((id, a, b), expected) in
                    spans(&p).iter().zip(row["sections"].as_array().unwrap())
                {
                    assert_eq!(id.0, expected[0].as_str().unwrap());
                    assert!(
                        (a - expected[1].as_f64().unwrap()).abs() < 1e-6
                            && (b - expected[2].as_f64().unwrap()).abs() < 1e-6,
                        "{name}"
                    );
                }
            }
        }
    }
    #[test]
    fn mixed_scene_range_moves_only_bare_scene_not_owned_sections() {
        let mut p = project();
        let mut bare = p.scenes[0].clone();
        bare.id = SceneId("own_bare".into());
        bare.start = time(5.);
        bare.sections.clear();
        p.scenes.push(bare);
        let selected = BTreeSet::from([p.scenes[0].id.clone(), SceneId("own_bare".into())]);
        let edits = move_mixed(&p, &[], &BTreeSet::new(), &selected, 5., 1., Snap::None);
        apply(&mut p, edits);
        assert_eq!(p.scenes[0].start, time(0.));
        assert_eq!(
            p.scene(&SceneId("own_bare".into())).unwrap().start,
            time(6.)
        );
    }
    #[test]
    fn duplicate_uses_global_block_gap_and_new_default_owners_then_undo() {
        let mut p = project();
        let before = p.clone();
        let ids = vec![
            SectionId("own_section_0".into()),
            SectionId("own_section_1".into()),
        ];
        let result = duplicate(&p, &ids, &BTreeSet::new());
        let copies = result.sections.clone();
        apply(&mut p, result.edits);
        let positions = spans(&p)
            .into_iter()
            .filter(|v| copies.contains(&v.0))
            .map(|(_, a, b)| (a, b))
            .collect::<Vec<_>>();
        assert_eq!(positions, vec![(4., 5.), (7., 8.)]);
        for scene in p
            .scenes
            .iter()
            .filter(|s| s.sections.iter().any(|s| copies.contains(&s.id)))
        {
            assert_eq!(scene.model, ModelChoice::ProjectDefault);
            assert!(scene.settings.is_empty() && scene.accepted_take.is_none());
        }
        let undo = editing::difference(&p, &before);
        apply(&mut p, undo);
        assert_eq!(spans(&p), spans(&before));
        assert_eq!(p.scenes.len(), 3);
        let mut p = project();
        p.scenes.remove(0);
        p.scenes[0].start = time(4.);
        p.scenes[0].duration = time(2.);
        p.scenes[0].sections[0].duration = time(2.);
        p.scenes[1].start = time(8.);
        p.scenes[1].duration = time(4.);
        p.scenes[1].sections[0].duration = time(4.);
        let result = duplicate(&p, &[SectionId("own_section_2".into())], &BTreeSet::new());
        let ids = result.sections.clone();
        apply(&mut p, result.edits);
        let copy = spans(&p).into_iter().find(|v| ids.contains(&v.0)).unwrap();
        assert_eq!((copy.1, copy.2), (0., 4.));
    }
    #[test]
    fn mixed_move_preserves_offsets_and_locked_audio_while_fit_is_conditional() {
        let mut p = project();
        let mut audio = AudioClip::new(
            AudioClipId("own_audio".into()),
            time(1.),
            time(1.),
            &p.audio,
        );
        let mut locked = audio.clone();
        locked.id = AudioClipId("own_locked".into());
        locked.locked = true;
        locked.start = time(5.);
        audio.lane = 1;
        p.audio_tracks.push(AudioTrack {
            id: AudioTrackId("own_track".into()),
            clips: vec![audio, locked],
        });
        let sections = vec![
            SectionId("own_section_0".into()),
            SectionId("own_section_1".into()),
        ];
        let audio = BTreeSet::from([
            AudioClipId("own_audio".into()),
            AudioClipId("own_locked".into()),
        ]);
        let commands = move_items(&p, &sections, &audio, 0., 100., Snap::Frames);
        apply(&mut p, commands);
        let positions = spans(&p);
        assert_eq!((positions[0].1, positions[1].1), (6., 9.));
        assert_eq!(
            seconds(
                p.audio_tracks[0]
                    .clips
                    .iter()
                    .find(|c| c.id.0 == "own_audio")
                    .unwrap()
                    .start
            ),
            7.
        );
        assert_eq!(
            seconds(
                p.audio_tracks[0]
                    .clips
                    .iter()
                    .find(|c| c.id.0 == "own_locked")
                    .unwrap()
                    .start
            ),
            5.
        );
        let mut p = project();
        assert!(!overflow(&p));
        assert!(fit(&p, true).is_empty());
        p.creative.timeline_duration = Some(time(8.));
        assert!(!can_fit_last(&p));
        assert!(fit(&p, false).is_empty());
        let commands = fit(&p, true);
        apply(&mut p, commands);
        assert!(!overflow(&p));
        assert_eq!(spans(&p).last().unwrap().2, 8.);
        let mut p = project();
        p.creative.timeline_duration = Some(time(10.5));
        assert!(can_fit_last(&p));
        let commands = fit(&p, false);
        apply(&mut p, commands);
        assert_eq!(spans(&p).last().unwrap().2, 10.5);
        assert_eq!(seconds(p.scenes.last().unwrap().duration), 0.5);
    }
}
