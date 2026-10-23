//! Director insertShotAfterCurrent geometry, without creating Scene metadata.
use super::{SceneId, Time, VideoProject, editing, group_commands};
use editing::{seconds, time};

/// Return (start, duration). An active Scene only searches forward; otherwise
/// the playhead is preferred and the first earlier free gap is a fallback.
pub(super) fn insertion(
    project: &VideoProject,
    current: Option<&SceneId>,
    playhead: f64,
    snap: editing::Snap,
) -> Option<(Time, Time)> {
    let limit = group_commands::duration(project);
    let duration = limit.min(1.);
    if duration <= 0. {
        return None;
    }
    let current = current.and_then(|id| project.scenes.iter().find(|scene| &scene.id == id));
    let preferred = current
        .map(|scene| seconds(scene.start) + seconds(scene.duration))
        .unwrap_or(playhead);
    let preferred = seconds(editing::snapped(preferred, snap, project.delivery.fps));
    let mut scenes = project.scenes.iter().collect::<Vec<_>>();
    scenes.sort_by(|a, b| {
        seconds(a.start)
            .total_cmp(&seconds(b.start))
            .then_with(|| {
                (seconds(a.start) + seconds(a.duration))
                    .total_cmp(&(seconds(b.start) + seconds(b.duration)))
            })
            .then_with(|| a.id.0.cmp(&b.id.0))
    });
    let mut gaps = vec![];
    let mut cursor: f64 = 0.;
    for scene in scenes {
        let start = seconds(scene.start);
        if start > cursor {
            gaps.push((cursor, start));
        }
        cursor = cursor.max(start + seconds(scene.duration));
    }
    if cursor < limit {
        gaps.push((cursor, limit));
    }
    let start = gaps
        .iter()
        .find_map(|(a, b)| {
            let start = a.max(preferred);
            (*b - start >= duration).then_some(start)
        })
        .or_else(|| {
            current
                .is_none()
                .then(|| {
                    gaps.iter()
                        .find_map(|(a, b)| (*b - *a >= duration).then_some(*a))
                })
                .flatten()
        })?;
    Some((time(start), time(duration)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use editing::Snap;

    fn project() -> VideoProject {
        let mut project = super::super::media_picker::tests::project();
        let template = project.scenes[0].clone();
        project.scenes.clear();
        project.creative.timeline_duration = Some(time(12.));
        for (id, start, end) in [("first", 0., 2.), ("middle", 4., 6.), ("last", 10., 12.)] {
            let mut scene = template.clone();
            scene.id = SceneId(id.into());
            scene.start = time(start);
            scene.duration = time(end - start);
            // Bare Scene intervals still occupy gaps independently of Sections.
            scene.sections.clear();
            project.scenes.push(scene);
        }
        project
    }

    #[test]
    fn current_scene_searches_forward_and_never_falls_back() {
        let project = project();
        assert_eq!(
            insertion(&project, Some(&SceneId("first".into())), 9., Snap::None),
            Some((time(2.), time(1.)))
        );
        assert_eq!(
            insertion(&project, Some(&SceneId("middle".into())), 0., Snap::None),
            Some((time(6.), time(1.)))
        );
        assert_eq!(
            insertion(&project, Some(&SceneId("last".into())), 0., Snap::None),
            None
        );
    }

    #[test]
    fn no_current_scene_uses_playhead_then_first_earlier_gap() {
        let project = project();
        assert_eq!(
            insertion(&project, None, 7., Snap::None),
            Some((time(7.), time(1.)))
        );
        assert_eq!(
            insertion(&project, None, 11., Snap::None),
            Some((time(2.), time(1.)))
        );
        assert_eq!(
            insertion(&project, Some(&SceneId("stale".into())), 11., Snap::None),
            Some((time(2.), time(1.)))
        );
    }

    #[test]
    fn snaps_preferred_start_and_preserves_subsecond_project_duration() {
        let mut project = project();
        project.scenes.clear();
        assert_eq!(
            insertion(&project, None, 1.49, Snap::Seconds),
            Some((time(1.), time(1.)))
        );
        assert_eq!(
            insertion(&project, None, 1.49, Snap::Frames),
            Some((time(1.5), time(1.)))
        );
        project.creative.timeline_duration = Some(time(0.5));
        assert_eq!(
            insertion(&project, None, 0., Snap::None),
            Some((time(0.), time(0.5)))
        );
        project.creative.timeline_duration = Some(time(0.));
        assert_eq!(insertion(&project, None, 0., Snap::None), None);
    }

    #[test]
    fn overlapping_scenes_do_not_create_false_gaps() {
        let mut project = project();
        project.scenes[0].duration = time(7.);
        assert_eq!(
            insertion(&project, None, 2., Snap::None),
            Some((time(7.), time(1.)))
        );
        project.scenes[0].duration = time(12.);
        assert_eq!(insertion(&project, None, 0., Snap::None), None);
    }
}
