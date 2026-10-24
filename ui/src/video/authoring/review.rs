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
/// A newly received `draft` with this side's edits to finished Scenes kept;
/// `received` becomes the server's own Scenes.
pub(in crate::app) fn merge_received(
    local: Option<&Draft>,
    received: &mut Vec<Option<StoryScene>>,
    mut draft: Draft,
) -> Draft {
    let server = draft.scenes.clone();
    if let Some(local) = local.filter(|l| l.id == draft.id && l.outline == draft.outline) {
        for (index, scene) in draft.scenes.iter_mut().enumerate() {
            let mine = local.scenes.get(index).and_then(Option::as_ref);
            let edited = mine.is_some() && mine != received.get(index).and_then(Option::as_ref);
            if edited && scene.is_some() {
                *scene = mine.cloned();
            }
        }
    }
    *received = server;
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
        let merged = merge_received(Some(&local), &mut received, server);
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
