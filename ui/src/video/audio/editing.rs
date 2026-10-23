use super::super::editing::{Snap, seconds, snapped, time};
use super::super::*;
pub(super) fn clip<'a>(project: &'a VideoProject, id: &AudioClipId) -> Option<&'a AudioClip> {
    project
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .find(|c| &c.id == id)
}
fn project_end(project: &VideoProject) -> f64 {
    project
        .scenes
        .iter()
        .map(|s| seconds(s.start) + seconds(s.duration))
        .chain(
            project
                .audio_tracks
                .iter()
                .flat_map(|t| &t.clips)
                .map(|c| seconds(c.start) + seconds(c.duration)),
        )
        .fold(10., f64::max)
}
pub(super) fn move_clips(
    project: &VideoProject,
    selected: &BTreeSet<AudioClipId>,
    delta: f64,
    snap: Snap,
) -> Vec<Edit> {
    let clips = project
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .filter(|c| selected.contains(&c.id) && !c.locked)
        .collect::<Vec<_>>();
    let anchor = clips
        .iter()
        .map(|c| seconds(c.start))
        .fold(f64::INFINITY, f64::min);
    if !anchor.is_finite() {
        return vec![];
    }
    let delta =
        (seconds(snapped(anchor + delta, snap, project.delivery.fps)) - anchor).max(-anchor);
    clips
        .into_iter()
        .map(|c| {
            let mut c = c.clone();
            c.start = time(seconds(c.start) + delta);
            Edit::UpdateAudioClip { clip: Box::new(c) }
        })
        .collect()
}
pub(super) fn resize_clip(
    project: &VideoProject,
    source: &AudioClip,
    end: bool,
    value: f64,
    snap: Snap,
) -> Vec<Edit> {
    if source.locked {
        return vec![];
    }
    let mut c = source.clone();
    let old_start = seconds(c.start);
    let old_end = old_start + seconds(c.duration);
    let at = seconds(snapped(value, snap, project.delivery.fps));
    if end {
        let new_end = at
            .max(old_start + 0.25)
            .min(project_end(project).max(old_start + 0.25));
        let source_end = c
            .source_out
            .map(seconds)
            .unwrap_or(seconds(c.source_in) + seconds(c.duration));
        c.source_out = Some(time(
            (source_end + new_end - old_end).max(seconds(c.source_in)),
        ));
        c.duration = time(new_end - old_start);
    } else {
        let start = at.clamp(0., (old_end - 0.25).max(0.));
        c.start = time(start);
        c.duration = time(old_end - start);
        c.source_in = time((seconds(c.source_in) + start - old_start).max(0.));
    }
    vec![Edit::UpdateAudioClip { clip: Box::new(c) }]
}
#[cfg(test)]
mod tests {
    use super::*;
    fn project() -> VideoProject {
        VideoProject::create(CreateProject {
            schema_version: 1,
            default_model: ModelSelection {
                adapter: "synthetic".into(),
                model: "synthetic".into(),
                variant: "test".into(),
                profile: "test".into(),
                profile_revision: "1".into(),
            },
            delivery: DeliverySettings {
                width: 64,
                height: 48,
                fps: time(24.),
            },
        })
        .unwrap()
    }
    fn apply(p: &mut VideoProject, commands: Vec<Edit>) {
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
    fn add(p: &mut VideoProject, id: &str, start: f64, duration: f64, locked: bool) {
        let mut clip = AudioClip::new(
            AudioClipId(id.into()),
            time(start),
            time(duration),
            &p.audio,
        );
        clip.locked = locked;
        apply(
            p,
            vec![Edit::AddAudioClip {
                track_id: AudioTrackId("audio_track_001".into()),
                clip: Box::new(clip),
            }],
        );
    }
    #[test]
    fn source_trim_matches_director_and_locked_gestures_do_nothing() {
        let mut p = project();
        add(&mut p, "a", 1., 3., false);
        let source = clip(&p, &AudioClipId("a".into())).unwrap();
        let commands = resize_clip(&p, source, false, 2., Snap::None);
        apply(&mut p, commands);
        let source = clip(&p, &AudioClipId("a".into())).unwrap();
        let commands = resize_clip(&p, source, true, 5., Snap::None);
        apply(&mut p, commands);
        let source = clip(&p, &AudioClipId("a".into())).unwrap();
        let oracle: Value = serde_json::from_str(include_str!(
            "../../../../assets/video/director-audio-defaults.json"
        ))
        .unwrap();
        assert_eq!(
            seconds(source.start),
            oracle["trim"]["start"].as_f64().unwrap()
        );
        assert_eq!(
            seconds(source.start) + seconds(source.duration),
            oracle["trim"]["end"].as_f64().unwrap()
        );
        assert_eq!(
            seconds(source.source_in),
            oracle["trim"]["source_in"].as_f64().unwrap()
        );
        assert_eq!(
            source.source_out.map(seconds),
            oracle["trim"]["source_out"].as_f64()
        );
        let mut locked = source.clone();
        locked.locked = true;
        assert!(resize_clip(&p, &locked, true, 8., Snap::None).is_empty());
    }
    #[test]
    fn group_move_preserves_offsets_skips_locks_and_undo_is_authoring_only() {
        let mut p = project();
        add(&mut p, "a", 1., 1., false);
        add(&mut p, "b", 2., 1., true);
        add(&mut p, "c", 3., 1., false);
        let before = p.clone();
        let selected = ["a", "b", "c"]
            .into_iter()
            .map(|id| AudioClipId(id.into()))
            .collect();
        let commands = move_clips(&p, &selected, -9., Snap::Frames);
        apply(&mut p, commands);
        assert_eq!(clip(&p, &AudioClipId("a".into())).unwrap().start, time(0.));
        assert_eq!(clip(&p, &AudioClipId("b".into())).unwrap().start, time(2.));
        assert_eq!(clip(&p, &AudioClipId("c".into())).unwrap().start, time(2.));
        let mut history = crate::app::timeline::editing::History::default();
        history.record(before.clone(), p.clone());
        let revision = p.project_revision;
        let commands = history.undo_commands(&p);
        apply(&mut p, commands);
        assert_eq!(p.audio_tracks, before.audio_tracks);
        assert!(p.project_revision > revision);
    }
}
