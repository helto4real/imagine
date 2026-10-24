//! Story review as in Director: the stage a proposal has reached and where
//! each Scene stands, with edits to finished Scenes kept while others expand.
use super::*;

/// Where one outline Scene stands in the storyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::app) enum SceneStatus {
    /// Planned only; not expanded yet.
    Outline,
    Queued,
    Generating,
    Ready,
    /// Expanded, then changed here.
    Edited,
    Failed,
}
impl SceneStatus {
    pub(in crate::app) fn label(self) -> &'static str {
        match self {
            Self::Outline => "Outline",
            Self::Queued => "Queued",
            Self::Generating => "Generating",
            Self::Ready => "Ready",
            Self::Edited => "Edited",
            Self::Failed => "Failed",
        }
    }
    fn color(self) -> egui::Color32 {
        match self {
            Self::Outline | Self::Queued => helto::DIM,
            Self::Generating => helto::FOCUS,
            Self::Ready => helto::OK,
            Self::Edited => helto::ACCENT,
            Self::Failed => helto::ERROR,
        }
    }
    /// Whether its fields can be edited while other Scenes still expand.
    pub(in crate::app) fn reviewable(self) -> bool {
        matches!(self, Self::Ready | Self::Edited)
    }
}
/// The status of outline Scene `index`; `received` holds the Scenes as the
/// server last returned them, so a difference is an edit made here.
pub(in crate::app) fn scene_status(
    draft: &Draft,
    received: &[Option<StoryScene>],
    index: usize,
) -> SceneStatus {
    if draft.failed_index == Some(index) {
        return SceneStatus::Failed;
    }
    match draft.scenes.get(index) {
        Some(Some(scene)) => {
            if received.get(index).and_then(Option::as_ref) == Some(scene) {
                SceneStatus::Ready
            } else {
                SceneStatus::Edited
            }
        }
        _ if draft.phase == Phase::Expanding => {
            let next = draft.scenes.iter().position(Option::is_none);
            if next == Some(index) {
                SceneStatus::Generating
            } else {
                SceneStatus::Queued
            }
        }
        _ => SceneStatus::Outline,
    }
}
/// The status of whole-Scene enhancement row `index`, in the Scene terms.
pub(in crate::app) fn row_status(
    draft: &Draft,
    received: &[Option<String>],
    index: usize,
) -> SceneStatus {
    let Some(row) = draft.fields.get(index) else {
        return SceneStatus::Queued;
    };
    if row.failure.is_some() {
        return SceneStatus::Failed;
    }
    match &row.text {
        Some(text) if received.get(index).and_then(Option::as_ref) == Some(text) => {
            SceneStatus::Ready
        }
        Some(_) => SceneStatus::Edited,
        None if draft.phase.busy()
            && draft
                .fields
                .iter()
                .position(|f| f.text.is_none() && f.failure.is_none())
                == Some(index) =>
        {
            SceneStatus::Generating
        }
        None => SceneStatus::Queued,
    }
}
/// A newly received `draft` with this side's edits to finished Scenes and
/// enhancement rows kept; `received` and `received_rows` become the server's.
pub(in crate::app) fn merge_received(
    local: Option<&Draft>,
    received: &mut Vec<Option<StoryScene>>,
    received_rows: &mut Vec<Option<String>>,
    mut draft: Draft,
) -> Draft {
    let server = draft.scenes.clone();
    let server_rows = draft.fields.iter().map(|f| f.text.clone()).collect();
    if let Some(local) = local.filter(|l| l.id == draft.id && l.outline == draft.outline) {
        for (index, scene) in draft.scenes.iter_mut().enumerate() {
            let mine = local.scenes.get(index).and_then(Option::as_ref);
            let edited = mine.is_some() && mine != received.get(index).and_then(Option::as_ref);
            if edited && scene.is_some() {
                *scene = mine.cloned();
            }
        }
        if local.fields.len() == draft.fields.len() {
            for (index, row) in draft.fields.iter_mut().enumerate() {
                let mine = &local.fields[index];
                row.include = mine.include;
                let edited = mine.text.is_some()
                    && mine.text.as_ref() != received_rows.get(index).and_then(Option::as_ref);
                if edited && row.text.is_some() {
                    row.text = mine.text.clone();
                }
            }
        }
    }
    *received = server;
    *received_rows = server_rows;
    draft
}
/// A small coloured status label.
pub(in crate::app) fn pill(ui: &mut egui::Ui, text: &str, color: egui::Color32) -> egui::Response {
    egui::Frame::new()
        .stroke(egui::Stroke::new(1., color.gamma_multiply(0.6)))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).small().color(color))
        })
        .inner
}
pub(in crate::app) fn status_pill(ui: &mut egui::Ui, status: SceneStatus) -> egui::Response {
    pill(ui, status.label(), status.color())
}
/// "Story brief · Outline · Storyboard": the active stage is gold and
/// finished stages are marked done.
pub(in crate::app) fn stage_strip(ui: &mut egui::Ui, draft: Option<&Draft>) {
    let expanded = draft.is_some_and(|d| d.scenes.iter().any(Option::is_some));
    let storyboard = draft.is_some_and(|d| {
        expanded || matches!(d.phase, Phase::Expanding) || d.failed_index.is_some()
    });
    let outline = draft.is_some_and(|d| d.outline.is_some());
    let active = if storyboard {
        2
    } else if outline {
        1
    } else {
        0
    };
    for (index, name) in ["Story brief", "Outline", "Storyboard"].iter().enumerate() {
        if index > 0 {
            ui.weak("·");
        }
        let done = index < active;
        let text = format!("{} {name}", index + 1);
        let text = if index == active {
            egui::RichText::new(text).color(helto::ACCENT).strong()
        } else if done {
            egui::RichText::new(format!("{text} ✓")).color(helto::TEXT)
        } else {
            egui::RichText::new(text).color(helto::DIM)
        };
        ui.label(text);
    }
}
/// A change to the outline's structure, as Director's outline cards offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::app) enum OutlineEdit {
    AddScene,
    /// Moves a Scene one place earlier (`false`) or later (`true`).
    MoveScene(usize, bool),
    RemoveScene(usize),
    AddSection(usize),
    MoveSection(usize, usize, bool),
    RemoveSection(usize, usize),
}
fn fresh_id(prefix: &str) -> String {
    format!(
        "{prefix}_{}",
        &uuid::Uuid::new_v4().simple().to_string()[..12]
    )
}
/// Whether the outline's structure can change: nothing expanded and nothing running.
pub(in crate::app) fn outline_editable(draft: &Draft) -> bool {
    draft.outline.is_some()
        && !draft.phase.busy()
        && draft.phase != Phase::Applied
        && draft.scenes.iter().all(Option::is_none)
}
/// Applies `edit` and returns the Scene to select. Limits as the server checks
/// them: 1–128 Scenes, 1–12 Sections each.
pub(in crate::app) fn edit_outline(draft: &mut Draft, edit: OutlineEdit, selected: usize) -> usize {
    let Some(outline) = &mut draft.outline else {
        return selected;
    };
    let scenes = &mut outline.scenes;
    let mut selected = selected.min(scenes.len().saturating_sub(1));
    match edit {
        OutlineEdit::AddScene if scenes.len() < 128 => {
            scenes.push(StoryScene {
                id: fresh_id("scene"),
                title: format!("Scene {}", scenes.len() + 1),
                summary: String::new(),
                description: String::new(),
                ambient_sound: String::new(),
                non_diegetic_music: String::new(),
                weight: 1.,
                boundary_from_previous: "hard_cut".into(),
                sections: vec![StorySection {
                    id: fresh_id("section"),
                    prompt: String::new(),
                    weight: 1.,
                    image_ids: vec![],
                    first_frame_id: String::new(),
                    last_frame_id: String::new(),
                    dialogue_intent: None,
                }],
                image_ids: vec![],
            });
            selected = scenes.len() - 1;
        }
        OutlineEdit::MoveScene(i, later) => {
            let j = if later { i + 1 } else { i.wrapping_sub(1) };
            if i < scenes.len() && j < scenes.len() {
                scenes.swap(i, j);
                selected = j;
            }
        }
        OutlineEdit::RemoveScene(i) if scenes.len() > 1 && i < scenes.len() => {
            scenes.remove(i);
            selected = i.min(scenes.len() - 1);
        }
        OutlineEdit::AddSection(i) => {
            if let Some(scene) = scenes.get_mut(i).filter(|s| s.sections.len() < 12) {
                scene.sections.push(StorySection {
                    id: fresh_id("section"),
                    prompt: String::new(),
                    weight: 1.,
                    image_ids: vec![],
                    first_frame_id: String::new(),
                    last_frame_id: String::new(),
                    dialogue_intent: None,
                });
            }
        }
        OutlineEdit::MoveSection(i, k, later) => {
            if let Some(scene) = scenes.get_mut(i) {
                let j = if later { k + 1 } else { k.wrapping_sub(1) };
                if k < scene.sections.len() && j < scene.sections.len() {
                    scene.sections.swap(k, j);
                }
            }
        }
        OutlineEdit::RemoveSection(i, k) => {
            if let Some(scene) = scenes
                .get_mut(i)
                .filter(|s| s.sections.len() > 1 && k < s.sections.len())
            {
                scene.sections.remove(k);
            }
        }
        _ => {}
    }
    draft.scenes = vec![None; scenes.len()];
    draft.failed_index = None;
    replan(draft);
    selected
}
/// The planned Scene lengths for the outline's current weights, or none when
/// they cannot fit the requested duration.
pub(in crate::app) fn replan(draft: &mut Draft) {
    let Some(outline) = &draft.outline else {
        return;
    };
    let weights = outline.scenes.iter().map(|s| s.weight).collect::<Vec<_>>();
    draft.planned_durations =
        durations(&weights, draft.request.duration_seconds, 15.).unwrap_or_default();
}
/// Why the outline cannot be expanded yet, as the server would refuse it.
pub(in crate::app) fn outline_blocker(draft: &Draft) -> Option<String> {
    let outline = draft.outline.as_ref()?;
    if outline
        .scenes
        .iter()
        .flat_map(|s| &s.sections)
        .any(|s| s.prompt.trim().is_empty())
    {
        return Some("Write a beat for every Section first.".into());
    }
    let weights = outline.scenes.iter().map(|s| s.weight).collect::<Vec<_>>();
    durations(&weights, draft.request.duration_seconds, 15.)
        .err()
        .map(|_| {
            format!(
                "{} Scenes cannot hold {:.0} s at 15 s each. Add a Scene.",
                outline.scenes.len(),
                draft.request.duration_seconds
            )
        })
}
/// Back to the outline: the storyboard is discarded, as any outline edit does.
pub(in crate::app) fn discard_storyboard(draft: &mut Draft) {
    draft.scenes = vec![None; draft.scenes.len()];
    draft.failed_index = None;
    draft.failure = None;
    draft.phase = Phase::Outline;
}
/// What a draft is doing, in words.
pub(in crate::app) fn phase_label(phase: Phase) -> &'static str {
    match phase {
        Phase::Inventory => "Reading images",
        Phase::Running => "Writing",
        Phase::Outline => "Outline ready",
        Phase::Expanding => "Expanding Scenes",
        Phase::Review => "Ready to apply",
        Phase::Failed => "Failed",
        Phase::Cancelled => "Cancelled",
        Phase::Applied => "Applied",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene(id: &str, summary: &str) -> StoryScene {
        StoryScene {
            id: id.into(),
            title: format!("Own {id}"),
            summary: summary.into(),
            description: String::new(),
            ambient_sound: String::new(),
            non_diegetic_music: String::new(),
            weight: 1.,
            boundary_from_previous: "hard_cut".into(),
            sections: vec![],
            image_ids: vec![],
        }
    }
    fn draft(scenes: Vec<Option<StoryScene>>, phase: Phase) -> Draft {
        serde_json::from_value(json!({
            "schema_version":1,"id":"own-draft","project_id":"own-project",
            "request":{"schema_version":1,"expected_project_revision":1,
                "provider_profile_id":"proxy","provider":"koboldcpp","prompt_profile_id":"normal",
                "operation":"outline","brief":"Own brief","duration_seconds":24.0},
            "phase":phase,"outline":{"scenes":[scene("a",""),scene("b",""),scene("c","")]},
            "scenes":scenes,"text":null,"failure":null,"failed_index":null,
            "prompt_profile_id":"normal","prompt_profile_revision":"1","profile_warning":"",
            "provider_snapshot":{},"planned_durations":[8.0,8.0,8.0]
        }))
        .unwrap()
    }
    #[test]
    fn scenes_are_ready_edited_generating_queued_or_failed() {
        let ready = scene("a", "Own arrival");
        let received = vec![Some(ready.clone()), None, None];
        let mut d = draft(vec![Some(ready.clone()), None, None], Phase::Expanding);
        let status = |d: &Draft, i| scene_status(d, &received, i);
        assert_eq!(status(&d, 0), SceneStatus::Ready);
        assert_eq!(status(&d, 1), SceneStatus::Generating);
        assert_eq!(status(&d, 2), SceneStatus::Queued);
        d.scenes[0].as_mut().unwrap().summary = "Own edited arrival".into();
        assert_eq!(status(&d, 0), SceneStatus::Edited);
        assert!(status(&d, 0).reviewable());
        d.phase = Phase::Failed;
        d.failed_index = Some(1);
        assert_eq!(status(&d, 1), SceneStatus::Failed);
        assert_eq!(status(&d, 2), SceneStatus::Outline);
    }
    #[test]
    fn outline_edits_keep_limits_and_unique_ids_and_replan_lengths() {
        let mut d = draft(vec![None, None, None], Phase::Outline);
        for s in &mut d.outline.as_mut().unwrap().scenes {
            s.sections.push(StorySection {
                id: format!("{}-1", s.id),
                prompt: "Own beat".into(),
                weight: 1.,
                image_ids: vec![],
                first_frame_id: String::new(),
                last_frame_id: String::new(),
                dialogue_intent: None,
            });
        }
        assert!(outline_editable(&d));
        let ids = |d: &Draft| {
            d.outline
                .as_ref()
                .unwrap()
                .scenes
                .iter()
                .map(|s| s.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(edit_outline(&mut d, OutlineEdit::MoveScene(0, true), 0), 1);
        assert_eq!(ids(&d), ["b", "a", "c"]);
        assert_eq!(edit_outline(&mut d, OutlineEdit::MoveScene(0, false), 0), 0);
        assert_eq!(
            ids(&d),
            ["b", "a", "c"],
            "the first Scene cannot move earlier"
        );
        assert_eq!(edit_outline(&mut d, OutlineEdit::AddScene, 0), 3);
        assert_eq!(d.scenes.len(), 4);
        assert_eq!(d.planned_durations, vec![6.; 4]);
        // A new Scene's Section has no beat yet, which blocks expansion.
        assert!(outline_blocker(&d).is_some());
        let new = d.outline.as_ref().unwrap().scenes[3].clone();
        assert!(!["a", "b", "c"].contains(&new.id.as_str()) && new.sections.len() == 1);
        d.outline.as_mut().unwrap().scenes[3].sections[0].prompt = "Own new beat".into();
        assert_eq!(outline_blocker(&d), None);
        edit_outline(&mut d, OutlineEdit::AddSection(3), 3);
        edit_outline(&mut d, OutlineEdit::MoveSection(3, 0, true), 3);
        let scene = &d.outline.as_ref().unwrap().scenes[3];
        assert_eq!(scene.sections[1].prompt, "Own new beat");
        edit_outline(&mut d, OutlineEdit::RemoveSection(3, 0), 3);
        edit_outline(&mut d, OutlineEdit::RemoveSection(3, 0), 3);
        assert_eq!(
            d.outline.as_ref().unwrap().scenes[3].sections.len(),
            1,
            "one Section stays"
        );
        assert_eq!(edit_outline(&mut d, OutlineEdit::RemoveScene(3), 3), 2);
        edit_outline(&mut d, OutlineEdit::RemoveScene(0), 0);
        // One Scene cannot hold the 24 s requested.
        edit_outline(&mut d, OutlineEdit::RemoveScene(0), 0);
        assert_eq!(d.outline.as_ref().unwrap().scenes.len(), 1);
        assert!(outline_blocker(&d).is_some_and(|b| b.contains("Add a Scene")));
        assert!(d.planned_durations.is_empty());
        edit_outline(&mut d, OutlineEdit::RemoveScene(0), 0);
        assert_eq!(
            d.outline.as_ref().unwrap().scenes.len(),
            1,
            "one Scene stays"
        );
        // Expanded Scenes lock the structure until the storyboard is discarded.
        d.scenes[0] = Some(d.outline.as_ref().unwrap().scenes[0].clone());
        assert!(!outline_editable(&d));
        discard_storyboard(&mut d);
        assert!(outline_editable(&d) && d.phase == Phase::Outline);
    }
    #[test]
    fn enhancement_rows_have_status_and_a_poll_keeps_their_edits_and_includes() {
        let row = |field, text: Option<&str>| FieldProposal {
            target: FieldTarget::Scene { field },
            original: "Own original".into(),
            text: text.map(Into::into),
            failure: None,
            include: true,
        };
        let mut local = draft(vec![None, None, None], Phase::Running);
        local.fields = vec![
            row(SceneField::Description, Some("Own proposal")),
            row(SceneField::Summary, None),
            row(SceneField::AmbientSound, None),
        ];
        let mut received_rows = vec![Some("Own proposal".to_owned()), None, None];
        assert_eq!(row_status(&local, &received_rows, 0), SceneStatus::Ready);
        assert_eq!(
            row_status(&local, &received_rows, 1),
            SceneStatus::Generating
        );
        assert_eq!(row_status(&local, &received_rows, 2), SceneStatus::Queued);
        local.fields[0].text = Some("Own edited proposal".into());
        local.fields[2].include = false;
        assert_eq!(row_status(&local, &received_rows, 0), SceneStatus::Edited);
        let mut server = local.clone();
        server.fields[0].text = Some("Own proposal".into());
        server.fields[1].text = Some("Own new summary".into());
        server.fields[2].include = true;
        server.fields[2].failure = Some("own_failure".into());
        let merged = merge_received(Some(&local), &mut vec![], &mut received_rows, server);
        assert_eq!(
            merged.fields[0].text.as_deref(),
            Some("Own edited proposal")
        );
        assert_eq!(merged.fields[1].text.as_deref(), Some("Own new summary"));
        assert!(!merged.fields[2].include, "the local Include choice stays");
        assert_eq!(row_status(&merged, &received_rows, 2), SceneStatus::Failed);
        assert_eq!(row_status(&merged, &received_rows, 1), SceneStatus::Ready);
    }
    #[test]
    fn a_poll_keeps_edits_to_finished_scenes_and_takes_new_ones() {
        let first = scene("a", "Own arrival");
        let mut received = vec![Some(first.clone()), None, None];
        let mut local = draft(vec![Some(first.clone()), None, None], Phase::Expanding);
        local.scenes[0].as_mut().unwrap().summary = "Own edited arrival".into();
        let second = scene("b", "Own signal");
        let server = draft(
            vec![Some(first.clone()), Some(second.clone()), None],
            Phase::Expanding,
        );
        let merged = merge_received(Some(&local), &mut received, &mut vec![], server);
        assert_eq!(
            merged.scenes[0].as_ref().unwrap().summary,
            "Own edited arrival"
        );
        assert_eq!(merged.scenes[1], Some(second.clone()));
        // The baseline is the server's, so the kept edit still reads as Edited.
        assert_eq!(received[0], Some(first));
        assert_eq!(scene_status(&merged, &received, 0), SceneStatus::Edited);
        assert_eq!(scene_status(&merged, &received, 1), SceneStatus::Ready);
    }
}
