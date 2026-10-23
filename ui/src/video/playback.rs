//! Draft playback: the viewer follows the playhead through each scene's
//! accepted Take or imported clip, and the playhead follows the viewer.
use super::editing::seconds;
use super::*;

#[derive(Default)]
pub(super) struct State {
    /// The scene whose media the viewer shows for the playhead.
    scene: Option<SceneId>,
    /// Continue into the next scene when this one ends.
    chain: bool,
    /// A cue asked for from the Timeline this frame: play or only show.
    pub request: Option<bool>,
    /// Applied once the cued media loads: its load nonce, offset and play.
    pending: Option<(u64, f64, bool)>,
}

/// The media that represents a scene in the draft: its accepted Take, else its clip.
fn scene_media(d: &Snapshot, scene: &Scene) -> Option<String> {
    if let Some(take) = &scene.accepted_take {
        return d.runtime_take(&take.id.0).map(str::to_owned);
    }
    let clip = d.project.scene_clip(&scene.id)?;
    d.runtime_asset(&clip.asset.0).map(str::to_owned)
}

impl ImagineApp {
    /// Shows the scene under the playhead at the matching time; `play` starts it
    /// and keeps playing through the following scenes.
    pub(super) fn timeline_cue(&mut self, ctx: &egui::Context, play: bool) {
        let Some(d) = self.timeline.document.clone() else {
            return;
        };
        let at = self.timeline.playhead;
        let Some(scene) = d.project.scenes.iter().find(|s| {
            let start = seconds(s.start);
            at >= start && at < start + seconds(s.duration)
        }) else {
            self.timeline.playback = State::default();
            return;
        };
        let offset = at - seconds(scene.start);
        let Some(media) = scene_media(&d, scene) else {
            let index = d.project.scenes.iter().position(|s| s.id == scene.id);
            self.timeline.playback = State::default();
            self.timeline.error = Some(format!(
                "Scene {} has no accepted Take to play.",
                index.map_or(0, |i| i + 1)
            ));
            return;
        };
        if self.timeline.preview_asset.as_deref() == Some(media.as_str())
            && let Some(player) = &mut self.timeline.player
        {
            player.seek(offset);
            if play && let Err(error) = player.start() {
                self.timeline.error = Some(error);
            }
        } else {
            self.timeline.selected_take = None;
            self.timeline_load_media(ctx, media);
            self.timeline.playback.pending = Some((self.timeline.load_nonce, offset, play));
        }
        self.timeline.playback.scene = Some(scene.id.clone());
        self.timeline.playback.chain = play;
    }
    /// Applies a cue's seek and play once its media has loaded.
    pub(super) fn timeline_cue_loaded(&mut self, nonce: u64) {
        let Some((pending, offset, play)) = self.timeline.playback.pending else {
            return;
        };
        if pending != nonce {
            return;
        }
        self.timeline.playback.pending = None;
        if let Some(player) = &mut self.timeline.player {
            player.seek(offset);
            if play && let Err(error) = player.start() {
                self.timeline.error = Some(error);
            }
        }
    }
    /// The Scene whose media the viewer shows: a chosen Take, else a Scene's
    /// accepted Take or clip.
    fn timeline_shown_scene(&self) -> Option<SceneId> {
        let d = self.timeline.document.as_ref()?;
        if let Some(take) = &self.timeline.selected_take {
            return d
                .view
                .takes
                .iter()
                .find(|t| &t.id.0 == take)
                .map(|t| t.scene_id.clone());
        }
        let shown = self.timeline.preview_asset.as_deref()?;
        d.project
            .scenes
            .iter()
            .find(|s| scene_media(d, s).as_deref() == Some(shown))
            .map(|s| s.id.clone())
    }
    /// Moves the playhead with the playing scene and continues into the next.
    /// Media played from the viewer moves it too, within its own Scene.
    pub(super) fn timeline_follow_playback(&mut self, ctx: &egui::Context) {
        if let Some(play) = self.timeline.playback.request.take() {
            self.timeline_cue(ctx, play);
        }
        let Some(scene_id) = self
            .timeline
            .playback
            .scene
            .clone()
            .or_else(|| self.timeline_shown_scene())
        else {
            return;
        };
        let Some(scene) = self
            .timeline
            .document
            .as_ref()
            .and_then(|d| d.project.scene(&scene_id).ok())
            .cloned()
        else {
            self.timeline.playback = State::default();
            return;
        };
        let Some(player) = &mut self.timeline.player else {
            return;
        };
        let (start, duration) = (seconds(scene.start), seconds(scene.duration));
        let position = player.position();
        if player.is_playing() && position < duration {
            self.timeline.playhead = start + position;
            ctx.request_repaint();
        } else if self.timeline.playback.chain && (player.finished() || position >= duration) {
            player.pause();
            let next = start + duration;
            if next < navigation::duration(&self.timeline.document.as_ref().unwrap().project) {
                self.timeline.playhead = next;
                self.timeline_cue(ctx, true);
            } else {
                self.timeline.playhead = next;
                self.timeline.playback.chain = false;
            }
        }
    }
    /// Media chosen outside the playhead ends draft playback.
    pub(super) fn timeline_end_playback(&mut self) {
        self.timeline.playback = State::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn app(ctx: &egui::Context, root: &std::path::Path) -> ImagineApp {
        let mut app = ImagineApp::synthetic_preview(ctx, root.into());
        app.client = None;
        let mut project = super::super::media_picker::tests::project();
        let mut second = project.scenes[0].clone();
        second.id = SceneId("own-second".into());
        second.start = project.scenes[0].duration;
        project.scenes.push(second);
        for scene in &mut project.scenes {
            scene.accepted_take = Some(AcceptedTake {
                id: TakeId(format!("take-{}", scene.id.0)),
                model: None,
                source_fingerprint: String::new(),
            });
        }
        let runtime: serde_json::Map<_, _> = project
            .scenes
            .iter()
            .map(|s| {
                (
                    format!("take-{}", s.id.0),
                    json!(format!("media-{}", s.id.0)),
                )
            })
            .collect();
        app.timeline.document = Some(Snapshot::with(project, json!({"runtime_takes":runtime})));
        app
    }
    #[test]
    fn the_viewer_follows_the_playhead_into_each_scene_and_back() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = app(&ctx, root.path());
        let first = app.timeline.document.as_ref().unwrap().project.scenes[0].clone();
        let media = format!("media-{}", first.id.0);
        // The first scene's Take is already shown: a cue seeks within it.
        app.timeline.player = Some(player::Player::new(player::Preview::synthetic(2.)));
        app.timeline.preview_asset = Some(media.clone());
        app.timeline.playhead = 0.5;
        app.timeline_cue(&ctx, false);
        assert_eq!(app.timeline.playback.scene, Some(first.id.clone()));
        assert!((app.timeline.player.as_ref().unwrap().position() - 0.5).abs() < 0.05);
        // Without a client the second scene cannot load; the cue is still recorded.
        app.timeline.playhead = seconds(first.duration) + 0.25;
        app.timeline_cue(&ctx, true);
        assert_eq!(
            app.timeline.playback.scene,
            Some(SceneId("own-second".into()))
        );
        assert!(app.timeline.playback.chain);
        // Choosing other media ends draft playback.
        app.timeline_end_playback();
        assert!(app.timeline.playback.scene.is_none());
        app.timeline.playhead = 99.;
        app.timeline_cue(&ctx, true);
        assert!(app.timeline.playback.scene.is_none() && !app.timeline.playback.chain);
    }
    #[test]
    fn a_take_played_from_the_viewer_moves_the_playhead_within_its_scene() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = app(&ctx, root.path());
        let second = app.timeline.document.as_ref().unwrap().project.scenes[1].clone();
        let mut document = app.timeline.document.take().unwrap();
        document.view = serde_json::from_value(json!({
            "takes":[{"id":"own-take","scene_id":second.id.0,"status":"candidate","provenance":null}]
        }))
        .unwrap();
        app.timeline.document = Some(document);
        app.timeline.selected_take = Some("own-take".into());
        let mut player = player::Player::new(player::Preview::synthetic(2.));
        player.seek(0.5);
        player.start().unwrap();
        app.timeline.player = Some(player);
        app.timeline.playhead = 0.;
        app.timeline_follow_playback(&ctx);
        let start = seconds(second.start);
        assert!(
            (app.timeline.playhead - (start + 0.5)).abs() < 0.1,
            "{} vs {start}",
            app.timeline.playhead
        );
        // Not draft playback: it does not continue into another Scene.
        assert!(!app.timeline.playback.chain);
    }
}
