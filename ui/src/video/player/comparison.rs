//! One wall clock, common-duration seek/loop and at most one audio child.
use super::*;
use imagine_video_contract::review::ReviewAction;

pub(in crate::app) struct Entry {
    pub id: String,
    pub label: String,
    pub accepted: bool,
    pub rejected: bool,
}
#[derive(Default)]
pub(in crate::app::timeline) struct Interaction {
    pub close: bool,
    pub select: Option<(usize, String)>,
    pub review: Option<(String, ReviewAction)>,
}
#[derive(Clone)]
pub(in crate::app) struct SelectionState {
    position: f64,
    audio_source: Option<usize>,
    loop_enabled: bool,
    loop_start: f64,
    loop_end: f64,
}
#[derive(Debug, Clone)]
struct Transport {
    position: f64,
    duration: f64,
    started: Option<(Instant, f64)>,
    loop_enabled: bool,
    loop_start: f64,
    loop_end: f64,
    loops: u64,
}
impl Transport {
    fn new(a: f64, b: f64) -> Result<Self, String> {
        let duration = a.min(b);
        if !a.is_finite() || !b.is_finite() || duration <= 0. {
            return Err("Both Takes need playable video media.".into());
        }
        Ok(Self {
            position: 0.,
            duration,
            started: None,
            loop_enabled: false,
            loop_start: 0.,
            loop_end: duration,
            loops: 0,
        })
    }
    fn advance(&mut self, now: Instant) -> bool {
        let Some((start, position)) = self.started else {
            return false;
        };
        self.position = position + now.duration_since(start).as_secs_f64();
        let end = if self.loop_enabled {
            self.loop_end
        } else {
            self.duration
        };
        if self.position >= end {
            if self.loop_enabled && end > self.loop_start {
                self.position = self.loop_start + (self.position - end) % (end - self.loop_start);
                self.started = Some((now, self.position));
                self.loops += 1;
                return true;
            }
            self.position = self.duration;
            self.started = None;
        }
        false
    }
    fn seek(&mut self, position: f64) {
        self.position = position.clamp(0., self.duration);
        if self.started.is_some() {
            self.started = Some((Instant::now(), self.position));
        }
    }
}
pub(in crate::app) struct Comparison {
    players: [Player; 2],
    clock: Transport,
    frame_limit: usize,
    audio: Option<super::super::super::video_ui::Audio>,
    audio_source: Option<usize>,
    pub(in crate::app::timeline) ids: [String; 2],
}
impl ImagineApp {
    pub(in crate::app::timeline) fn timeline_compare(
        &mut self,
        ctx: &egui::Context,
        document: &Snapshot,
    ) {
        let selection_state = self
            .timeline
            .comparison
            .as_mut()
            .map(|c| c.selection_state())
            .or_else(|| self.timeline.comparison_state.clone());
        self.pause_timeline();
        let [Some(a), Some(b)] = self.timeline.comparison_selection.clone() else {
            return;
        };
        if a == b {
            return;
        }
        let Some(scene) = self.timeline.scene() else {
            return;
        };
        if [&a, &b].into_iter().any(|id| {
            !document.view.takes.iter().any(|t| {
                t.id.0 == *id
                    && t.scene_id == scene.id
                    && super::super::review::attached(document, t)
            })
        }) {
            return;
        }
        let sources = [document.runtime_take(&a), document.runtime_take(&b)];
        let [Some(source_a), Some(source_b)] = sources else {
            self.timeline.error = Some(
                "Both Take media sources must be available. Metadata remains available.".into(),
            );
            return;
        };
        let sources = [source_a.to_owned(), source_b.to_owned()];
        let ids = [a, b];
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.clear_preview();
        self.timeline.comparison_state = selection_state.clone();
        self.timeline.load_nonce += 1;
        let nonce = self.timeline.load_nonce;
        let epoch = self.timeline.epoch;
        let cancel = Arc::new(AtomicBool::new(false));
        self.timeline.loading = Some((nonce, cancel.clone()));
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(async {
                let mut previews = vec![];
                for (side, asset) in sources.into_iter().enumerate() {
                    let unavailable = |message: String| {
                        format!(
                            "Take {} preview unavailable: {message}",
                            if side == 0 { "A" } else { "B" }
                        )
                    };
                    let info = client
                        .timeline_request(
                            &session,
                            false,
                            &format!("/v1/video/assets/{asset}"),
                            None,
                        )
                        .await
                        .map_err(|e| unavailable(e.message))?;
                    let bytes = client
                        .timeline_media(&session, &asset)
                        .await
                        .map_err(|e| unavailable(e.message))?;
                    previews.push(decode(bytes, info, cancel.clone()).map_err(unavailable)?);
                }
                let preview_b = previews.pop().unwrap();
                let preview_a = previews.pop().unwrap();
                let mut comparison = Comparison::new(preview_a, preview_b, ids)?;
                if let Some(state) = selection_state {
                    comparison.restore_selection(state);
                }
                Ok(Box::new(comparison))
            });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Comparison {
                    epoch,
                    nonce,
                    result,
                }),
            );
        });
    }
}
fn control(ui: &egui::Ui, name: &str, r: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::super::qa::control(ui, name.into(), r);
    let _ = (ui, name, r);
}
impl Comparison {
    fn selection_state(&mut self) -> SelectionState {
        self.clock.advance(Instant::now());
        SelectionState {
            position: self.clock.position,
            audio_source: self.audio_source,
            loop_enabled: self.clock.loop_enabled,
            loop_start: self.clock.loop_start,
            loop_end: self.clock.loop_end,
        }
    }
    fn restore_selection(&mut self, state: SelectionState) {
        self.clock.seek(state.position);
        self.audio_source = state.audio_source;
        self.clock.loop_enabled = state.loop_enabled;
        self.clock.loop_start = state
            .loop_start
            .clamp(0., (self.clock.duration - 1. / 24.).max(0.));
        self.clock.loop_end = state.loop_end.clamp(
            (self.clock.loop_start + 1. / 24.).min(self.clock.duration),
            self.clock.duration,
        );
    }
    fn swap(&mut self) -> Result<(), String> {
        self.players.swap(0, 1);
        self.ids.swap(0, 1);
        // Audio A/B names a side. Replace the previous child before playing
        // that side's newly selected PCM at the unchanged shared position.
        self.restart_audio()
    }
    pub(in crate::app::timeline) fn new(
        a: Preview,
        b: Preview,
        ids: [String; 2],
    ) -> Result<Self, String> {
        if a.frames.is_empty() || b.frames.is_empty() {
            return Err("Both Takes need playable video media.".into());
        }
        let clock = Transport::new(a.duration, b.duration)?;
        let frame_limit = a.frames.len().min(b.frames.len()) - 1;
        Ok(Self {
            players: [Player::new(a), Player::new(b)],
            clock,
            frame_limit,
            audio: None,
            audio_source: None,
            ids,
        })
    }
    pub(in crate::app::timeline) fn pause(&mut self) {
        self.clock.advance(Instant::now());
        self.clock.started = None;
        self.audio = None;
    }
    fn restart_audio(&mut self) -> Result<(), String> {
        self.audio = None;
        if self.clock.started.is_some()
            && let Some(pcm) = self.audio_window()
        {
            self.audio = Some(
                super::super::super::video_ui::Audio::start(pcm.to_vec()).map_err(str::to_owned)?,
            );
        }
        Ok(())
    }
    fn audio_window(&self) -> Option<&[u8]> {
        let pcm = &self.players[self.audio_source?].media.pcm;
        let start = (self.clock.position * 32000.).floor() as usize * 8;
        let end = ((self.clock.duration * 32000.).floor() as usize * 8).min(pcm.len());
        (start < end).then(|| &pcm[start..end])
    }
    fn play(&mut self) -> Result<(), String> {
        let end = if self.clock.loop_enabled {
            self.clock.loop_end
        } else {
            self.clock.duration
        };
        if self.clock.position >= end {
            self.clock.seek(self.clock.loop_start);
        }
        self.clock.started = Some((Instant::now(), self.clock.position));
        if let Err(error) = self.restart_audio() {
            self.pause();
            return Err(error);
        }
        Ok(())
    }
    fn current_frame(&self) -> usize {
        ((self.clock.position * 24.).floor() as usize).min(self.frame_limit)
    }
    pub(in crate::app::timeline) fn show(
        &mut self,
        ui: &mut egui::Ui,
        privacy: bool,
        entries: &[Entry],
        enabled: bool,
    ) -> Result<Interaction, String> {
        let looped = self.clock.advance(Instant::now());
        if self.clock.started.is_none() {
            self.audio = None;
        } else if looped && let Err(error) = self.restart_audio() {
            self.pause();
            return Err(error);
        }
        let revealed = helto::revealed(ui, ui.available_rect_before_wrap(), privacy);
        if !revealed {
            self.pause();
        }
        let mut interaction = Interaction::default();
        let mut error = None;
        ui.horizontal_wrapped(|ui| {
            ui.strong("A/B Take review");
            let r = ui
                .add_enabled_ui(revealed, |ui| {
                    helto::icon_button(ui, crate::icons::Icon::Swap, "Swap A/B", false)
                })
                .inner;
            control(ui, "comparison-swap", &r);
            if r.clicked()
                && let Err(e) = self.swap()
            {
                error = Some(e);
            }
            let r = helto::icon_button(ui, crate::icons::Icon::Close, "Close comparison", false);
            control(ui, "comparison-close", &r);
            interaction.close = r.clicked();
            if !revealed {
                ui.weak("Hidden comparison");
            }
        });
        let images = ui.available_rect_before_wrap();
        let height = (images.height() - 96.).max(24.);
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(images.width(), height), egui::Sense::hover());
        let gap = 6.;
        let half = (rect.width() - gap) / 2.;
        let vertical = rect.width() < 400.;
        let frame = self.current_frame();
        for side in 0..2 {
            let panel = if vertical {
                let height = (rect.height() - gap) / 2.;
                egui::Rect::from_min_size(
                    rect.min + egui::vec2(0., side as f32 * (height + gap)),
                    egui::vec2(rect.width(), height),
                )
            } else {
                egui::Rect::from_min_size(
                    rect.min + egui::vec2(side as f32 * (half + gap), 0.),
                    egui::vec2(half, rect.height()),
                )
            };
            let header = egui::Rect::from_min_max(
                panel.min,
                egui::pos2(panel.right(), (panel.top() + 68.).min(panel.bottom())),
            );
            let mut side_ui = ui.new_child(
                egui::UiBuilder::new()
                    .id_salt(("comparison-side", side))
                    .max_rect(header),
            );
            side_ui.set_clip_rect(header.intersect(ui.clip_rect()));
            let entry = entries.iter().find(|entry| entry.id == self.ids[side]);
            side_ui.horizontal(|ui| {
                ui.strong(if side == 0 { "A" } else { "B" });
                ui.add_enabled_ui(revealed && enabled, |ui| {
                    for (key, icon, label, action, available) in [
                        (
                            "accept",
                            crate::icons::Icon::Check,
                            "Accept Take",
                            ReviewAction::Accept,
                            !entry.is_some_and(|e| e.accepted),
                        ),
                        (
                            "reject",
                            crate::icons::Icon::Close,
                            "Reject Take",
                            ReviewAction::Reject,
                            !entry.is_some_and(|e| e.rejected),
                        ),
                    ] {
                        let r = ui
                            .add_enabled_ui(available, |ui| {
                                helto::icon_button(ui, icon, label, false)
                            })
                            .inner;
                        control(ui, &format!("comparison-{side}-{key}"), &r);
                        if r.clicked() {
                            interaction.review = Some((self.ids[side].clone(), action));
                        }
                    }
                });
            });
            if revealed {
                let mut picked = self.ids[side].clone();
                let r = egui::ComboBox::from_id_salt(("comparison-take", side))
                    .width((header.width() - 8.).max(20.))
                    .selected_text(entry.map_or("Unavailable", |e| e.label.as_str()))
                    .show_ui(&mut side_ui, |ui| {
                        for candidate in entries {
                            let r = ui.selectable_value(
                                &mut picked,
                                candidate.id.clone(),
                                &candidate.label,
                            );
                            control(ui, &format!("comparison-{side}-take-{}", candidate.id), &r);
                        }
                    })
                    .response;
                control(&side_ui, &format!("comparison-{side}-take"), &r);
                if picked != self.ids[side] {
                    interaction.select = Some((side, picked));
                }
            } else {
                side_ui.weak("Hidden Take");
            }
            if revealed {
                self.players[side].frame = frame;
                let picture =
                    egui::Rect::from_min_max(egui::pos2(panel.left(), header.bottom()), panel.max);
                if picture.height() > 0. {
                    self.players[side].paint_frame(ui, picture);
                }
            }
        }
        ui.add_enabled_ui(revealed, |ui| {
            ui.horizontal_wrapped(|ui| {
                let r = if self.clock.started.is_some() {
                    helto::icon_button(ui, crate::icons::Icon::Pause, "Pause", false)
                } else {
                    helto::icon_button(ui, crate::icons::Icon::Play, "Play", false)
                };
                control(ui, "comparison-play", &r);
                if r.clicked() {
                    if self.clock.started.is_some() {
                        self.pause();
                    } else if let Err(e) = self.play() {
                        error = Some(e);
                    }
                }
                let mut position = self.clock.position;
                let r = ui.add(
                    egui::Slider::new(&mut position, 0.0..=self.clock.duration).show_value(false),
                );
                control(ui, "comparison-seek", &r);
                if r.changed() {
                    self.clock.seek(position);
                    if let Err(e) = self.restart_audio() {
                        error = Some(e);
                    }
                }
                ui.weak(format!(
                    "{:.2} / {:.2} s",
                    self.clock.position, self.clock.duration
                ));
                for (side, label, name) in [
                    (None, "Muted", "comparison-muted"),
                    (Some(0), "Audio A", "comparison-audio-a"),
                    (Some(1), "Audio B", "comparison-audio-b"),
                ] {
                    let r = ui.selectable_label(self.audio_source == side, label);
                    control(ui, name, &r);
                    if r.clicked() {
                        self.audio_source = side;
                        if let Err(e) = self.restart_audio() {
                            error = Some(e);
                        }
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                let r = ui.checkbox(&mut self.clock.loop_enabled, "Loop");
                control(ui, "comparison-loop", &r);
                if r.changed() {
                    self.clock.seek(self.clock.position);
                }
                let before = (self.clock.loop_start, self.clock.loop_end);
                ui.label("From");
                let r = ui.add(
                    egui::DragValue::new(&mut self.clock.loop_start)
                        .range(0.0..=self.clock.duration)
                        .suffix(" s"),
                );
                control(ui, "comparison-loop-start", &r);
                ui.label("To");
                let r = ui.add(
                    egui::DragValue::new(&mut self.clock.loop_end)
                        .range(0.0..=self.clock.duration)
                        .suffix(" s"),
                );
                control(ui, "comparison-loop-end", &r);
                self.clock.loop_start = self
                    .clock
                    .loop_start
                    .clamp(0., (self.clock.duration - 1. / 24.).max(0.));
                self.clock.loop_end = self.clock.loop_end.clamp(
                    (self.clock.loop_start + 1. / 24.).min(self.clock.duration),
                    self.clock.duration,
                );
                if before != (self.clock.loop_start, self.clock.loop_end) {
                    self.clock.seek(self.clock.position);
                }
            });
        });
        if let Some(error) = error {
            self.pause();
            return Err(error);
        }
        if self.clock.started.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(15));
        }
        Ok(interaction)
    }
    #[cfg(feature = "ui-preview")]
    pub(in crate::app::timeline) fn snapshot(&mut self) -> Value {
        json!({"ids":self.ids,"position":self.clock.position,"duration":self.clock.duration,"playing":self.clock.started.is_some(),"frames":[self.players[0].frame,self.players[1].frame],"frame_limit":self.frame_limit,"audio":self.audio_source.map(|s|if s==0{"A"}else{"B"}).unwrap_or("muted"),"audio_child":self.audio.as_mut().is_some_and(|a|a.is_running()),"loops":self.clock.loops,"loop_enabled":self.clock.loop_enabled})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn swap_preserves_clock_and_retargets_side_audio_then_new_pair_clamps_state() {
        let media = |duration, value| Preview {
            width: 2,
            height: 2,
            frames: vec![vec![0; 12]; 48],
            pcm: vec![value; 128000]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            duration,
        };
        let mut c =
            Comparison::new(media(2., 0.25), media(2., 0.75), ["a".into(), "b".into()]).unwrap();
        c.clock.seek(0.75);
        c.clock.loop_enabled = true;
        c.clock.loop_start = 0.5;
        c.clock.loop_end = 1.5;
        c.audio_source = Some(0);
        c.swap().unwrap();
        assert_eq!(c.ids, ["b", "a"]);
        assert_eq!(c.clock.position, 0.75);
        assert_eq!(c.audio_source, Some(0));
        assert_eq!(
            f32::from_le_bytes(c.audio_window().unwrap()[..4].try_into().unwrap()),
            0.75
        );
        let mut next =
            Comparison::new(media(0.6, 0.5), media(1., 0.25), ["c".into(), "a".into()]).unwrap();
        next.restore_selection(c.selection_state());
        assert_eq!(next.clock.position, 0.6);
        assert!(next.clock.started.is_none() && next.audio.is_none());
        assert_eq!(next.audio_source, Some(0));
        assert_eq!((next.clock.loop_start, next.clock.loop_end), (0.5, 0.6));
    }
    #[test]
    fn selected_pcm_source_and_seek_window_are_exact_and_stop_at_common_end() {
        let media = |duration, pcm| Preview {
            width: 2,
            height: 2,
            frames: vec![vec![0; 12]; 48],
            pcm,
            duration,
        };
        let a = (0..128000)
            .flat_map(|v| (v as f32 / 100000.).to_le_bytes())
            .collect();
        let b = vec![0.25f32; 64000]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut c = Comparison::new(media(2., a), media(1., b), ["a".into(), "b".into()]).unwrap();
        c.clock.seek(0.25);
        assert!(c.audio_window().is_none());
        c.audio_source = Some(0);
        let pcm = c.audio_window().unwrap();
        assert_eq!(pcm.len(), 24000 * 8);
        assert_eq!(f32::from_le_bytes(pcm[..4].try_into().unwrap()), 0.16);
        c.audio_source = Some(1);
        assert_eq!(
            f32::from_le_bytes(c.audio_window().unwrap()[..4].try_into().unwrap()),
            0.25
        );
        c.clock.seek(1.);
        assert!(c.audio_window().is_none());
    }
    #[test]
    fn rounded_container_duration_cannot_advance_one_side_past_common_last_frame() {
        let media = |count, duration| Preview {
            width: 2,
            height: 2,
            frames: vec![vec![0; 12]; count],
            pcm: vec![],
            duration,
        };
        let mut comparison = Comparison::new(
            media(124, 5.166667),
            media(73, 3.041667),
            ["a".into(), "b".into()],
        )
        .unwrap();
        comparison.clock.seek(comparison.clock.duration);
        assert_eq!(comparison.current_frame(), 72);
    }
    #[test]
    fn common_duration_loop_seek_and_end_share_one_position() {
        let mut t = Transport::new(4., 6.).unwrap();
        assert_eq!(t.duration, 4.);
        t.seek(9.);
        assert_eq!(t.position, 4.);
        let now = Instant::now();
        t.started = Some((now, 0.));
        t.advance(now + Duration::from_secs(5));
        assert_eq!(t.position, 4.);
        assert!(t.started.is_none());
        t.loop_enabled = true;
        t.loop_start = 1.;
        t.loop_end = 2.;
        t.started = Some((now, 1.));
        assert!(t.advance(now + Duration::from_millis(1250)));
        assert_eq!(t.position, 1.25);
        assert_eq!(t.loops, 1);
        assert!(Transport::new(0., 2.).is_err());
    }
}
