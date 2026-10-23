//! Native audio authoring state; pure gestures and views have separate owners.
use super::editing::{seconds, time};
use super::*;
mod editing;
mod inspector;
mod tracks;
use editing::clip;
impl Timeline {
    pub(super) fn add_audio(&mut self, project: &VideoProject, asset: Option<AssetId>) -> Edit {
        let start = self
            .scene()
            .map(|s| s.start)
            .unwrap_or(Time { num: 0, den: 1 });
        let id = AudioClipId(format!("audio_clip_{}", uuid::Uuid::new_v4().simple()));
        let mut c = AudioClip::new(id.clone(), start, time(1.), &project.audio);
        c.asset = asset;
        self.audio_selected = BTreeSet::from([id.clone()]);
        self.audio_primary = Some(id);
        self.section = None;
        self.project_audio = false;
        self.audio_draft = Some(c.clone());
        Edit::AddAudioClip {
            track_id: project
                .audio_tracks
                .first()
                .map(|t| t.id.clone())
                .unwrap_or_else(|| AudioTrackId("audio_track_001".into())),
            clip: Box::new(c),
        }
    }
    pub(super) fn clear_audio_selection(&mut self) {
        self.audio_selected.clear();
        self.audio_primary = None;
        self.audio_draft = None;
        self.project_audio = false;
    }
}
