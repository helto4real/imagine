//! One wall clock, common-duration seek/loop and at most one audio child.
use super::*;
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
            !document.raw["takes"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|t| t["id"].as_str() == Some(id) && t["scene_id"] == json!(scene.id))
        }) {
            return;
        }
        let sources = [
            document.raw["runtime_takes"][&a].as_str(),
            document.raw["runtime_takes"][&b].as_str(),
        ];
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
                for asset in sources {
                    let info = client
                        .timeline_request(
                            &session,
                            false,
                            &format!("/v1/video/assets/{asset}"),
                            None,
                        )
                        .await
                        .map_err(|e| e.message)?;
                    let bytes = client
                        .timeline_media(&session, &asset)
                        .await
                        .map_err(|e| e.message)?;
                    previews.push(decode(bytes, info, cancel.clone())?);
                }
                let preview_b = previews.pop().unwrap();
                let preview_a = previews.pop().unwrap();
                Comparison::new(preview_a, preview_b, ids).map(Box::new)
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
    ) -> Result<bool, String> {
        let looped = self.clock.advance(Instant::now());
        if self.clock.started.is_none() {
            self.audio = None;
        } else if looped && let Err(error) = self.restart_audio() {
            self.pause();
            return Err(error);
        }
        let boundary = ui.available_rect_before_wrap().intersect(ui.clip_rect());
        let revealed = !privacy || ui.input(|i| i.focused) && ui.rect_contains_pointer(boundary);
        if !revealed {
            self.pause();
        }
        let mut close = false;
        let mut error = None;
        ui.horizontal_wrapped(|ui| {
            ui.strong("A/B Take review");
            let r = ui.button("Close comparison");
            control(ui, "comparison-close", &r);
            close = r.clicked();
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
        let frame = self.current_frame();
        for side in 0..2 {
            let panel = egui::Rect::from_min_size(
                rect.min + egui::vec2(side as f32 * (half + gap), 0.),
                egui::vec2(half, rect.height()),
            );
            ui.painter().text(
                panel.left_top() + egui::vec2(5., 4.),
                egui::Align2::LEFT_TOP,
                if side == 0 { "A" } else { "B" },
                egui::FontId::proportional(14.),
                helto::TEXT,
            );
            if revealed {
                self.players[side].frame = frame;
                self.players[side].paint_frame(ui, panel.shrink2(egui::vec2(0., 22.)));
            }
        }
        ui.add_enabled_ui(revealed, |ui| {
            ui.horizontal_wrapped(|ui| {
                let r = ui.button(if self.clock.started.is_some() {
                    "Pause"
                } else {
                    "Play"
                });
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
        Ok(close)
    }
    #[cfg(feature = "ui-preview")]
    pub(in crate::app::timeline) fn snapshot(&mut self) -> Value {
        json!({"position":self.clock.position,"duration":self.clock.duration,"playing":self.clock.started.is_some(),"frames":[self.players[0].frame,self.players[1].frame],"frame_limit":self.frame_limit,"audio":self.audio_source.map(|s|if s==0{"A"}else{"B"}).unwrap_or("muted"),"audio_child":self.audio.as_mut().is_some_and(|a|a.is_running()),"loops":self.clock.loops,"loop_enabled":self.clock.loop_enabled})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
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
