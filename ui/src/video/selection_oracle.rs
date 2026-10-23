//! Direct comparisons with the source-executed Director G8 oracle.
use super::*;
use editing::{Snap, time};

fn project(spans: &[(f64, f64)]) -> VideoProject {
    let mut project = media_picker::tests::project();
    let template = project.scenes[0].clone();
    project.scenes.clear();
    project.creative.timeline_duration = Some(time(12.));
    for (index, &(start, end)) in spans.iter().enumerate() {
        let mut scene = template.clone();
        scene.id = SceneId(format!("own_scene_{index}"));
        scene.start = time(start);
        scene.duration = time(end - start);
        scene.sections[0].id = SectionId(format!("own_section_{index}"));
        scene.sections[0].start = time(0.);
        scene.sections[0].duration = time(end - start);
        project.scenes.push(scene);
    }
    project
}

fn timeline(project: &VideoProject) -> Timeline {
    let mut timeline = Timeline::default();
    timeline.document = Some(Snapshot::with(project.clone(), json!({})));
    timeline
}

fn item(id: &str) -> selection::Item {
    if id.starts_with("own_section_") {
        selection::Item::Section(SectionId(id.into()))
    } else if id.starts_with("own_audio_") {
        selection::Item::Audio(AudioClipId(id.into()))
    } else {
        selection::Item::Scene(SceneId(id.into()))
    }
}

fn item_id(item: selection::Item) -> String {
    match item {
        selection::Item::Scene(id) => id.0,
        selection::Item::Section(id) => id.0,
        selection::Item::Audio(id) => id.0,
    }
}

#[test]
fn ordered_timeline_selection_matches_director_actions() {
    let oracle: Value = serde_json::from_str(include_str!("group8-director.json")).unwrap();
    let mut cases = 0;
    for row in oracle["cases"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        if !name.starts_with("selection_") {
            continue;
        }
        cases += 1;
        let mut project = project(&[(0., 1.), (3., 4.), (10., 11.)]);
        let mut bare = project.scenes[0].clone();
        bare.id = SceneId("own_bare".into());
        bare.start = time(6.);
        bare.duration = time(1.);
        bare.sections.clear();
        project.scenes.push(bare);
        project.audio_tracks = vec![AudioTrack {
            id: AudioTrackId("own_track".into()),
            clips: (0..3)
                .map(|lane| {
                    let mut clip = AudioClip::new(
                        AudioClipId(format!("own_audio_{lane}")),
                        time(0.),
                        time(f64::from(5 - lane)),
                        &project.audio,
                    );
                    clip.lane = lane;
                    clip
                })
                .collect(),
        }];
        let mut timeline = timeline(&project);
        for action in row["actions"].as_array().unwrap() {
            let operation = action[0].as_str().unwrap();
            let target = item(action[1].as_str().unwrap());
            let (toggle, range) = match operation {
                "select" => (false, false),
                "toggle" => (true, false),
                "range" => (false, true),
                _ => panic!("Unknown Director selection action: {operation}"),
            };
            timeline.pick_item(&project, target, toggle, range, false);
        }
        let actual = timeline
            .selection_items(&project)
            .into_iter()
            .map(item_id)
            .collect::<Vec<_>>();
        let expected = row["selection"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{name}");
    }
    assert!(cases > 0, "Director oracle has no selection cases");
}

#[test]
fn scene_insertion_from_actual_primary_matches_director() {
    let oracle: Value = serde_json::from_str(include_str!("group8-director.json")).unwrap();
    let mut cases = 0;
    for row in oracle["cases"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        if !name.starts_with("scene_add_") {
            continue;
        }
        cases += 1;
        let project = project(&[(0., 2.), (4., 6.), (10., 12.)]);
        let mut timeline = timeline(&project);
        if let Some(id) = row["selected"].as_str() {
            timeline.pick_item(&project, item(id), false, false, false);
        }
        let current = timeline.primary_scene(&project);
        let actual = scene_commands::insertion(
            &project,
            current.as_ref(),
            row["playhead"].as_f64().unwrap(),
            Snap::None,
        );
        let expected = row["added"].as_array().map(|added| {
            let start = added[0].as_f64().unwrap();
            let end = added[1].as_f64().unwrap();
            (time(start), time(end - start))
        });
        assert_eq!(actual, expected, "{name}");
    }
    assert!(cases > 0, "Director oracle has no Scene insertion cases");
}
