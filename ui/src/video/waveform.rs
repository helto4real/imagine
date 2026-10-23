//! RAM-only waveform requests and Director source-range/volume projection.
use super::editing::seconds;
use super::*;
type Key = (AssetId, u64, usize);
#[derive(Default)]
pub(super) struct State {
    pub project: Option<ProjectId>,
    pub values: std::collections::BTreeMap<Key, Result<Waveform, String>>,
    pub loading: BTreeSet<Key>,
}
#[derive(Clone, serde::Deserialize)]
pub(super) struct Waveform {
    pub duration_seconds: f64,
    pub peaks: Vec<f64>,
}
pub(super) fn count(width: f32) -> usize {
    ((width / 2.).ceil() as usize).clamp(16, 512)
}
fn range(wave: &Waveform, clip: &AudioClip) -> (usize, usize) {
    let n = wave.peaks.len();
    if n == 0 {
        return (0, 0);
    }
    let duration = wave.duration_seconds;
    if !duration.is_finite() || duration <= 0. {
        return (0, n);
    }
    let start = seconds(clip.source_in).clamp(0., duration);
    let end = clip
        .source_out
        .map(seconds)
        .unwrap_or(start + seconds(clip.duration))
        .clamp(start, duration);
    let a = ((start / duration * n as f64).floor() as usize).min(n - 1);
    let b = ((end / duration * n as f64).ceil() as usize).clamp(a + 1, n);
    (a, b)
}
pub(super) fn bars(wave: &Waveform, clip: &AudioClip, n: usize) -> Vec<f64> {
    let (a, b) = range(wave, clip);
    let input = &wave.peaks[a..b];
    if input.is_empty() {
        return vec![];
    }
    let gain = clip.volume.max(0.) / 100.;
    (0..n)
        .map(|i| {
            let a = i * input.len() / n;
            let b = ((i + 1) * input.len())
                .div_ceil(n)
                .max(a + 1)
                .min(input.len());
            let v = input[a..b]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if gain == 1. {
                v
            } else {
                (v * gain).clamp(0., 1.)
            }
        })
        .collect()
}
fn detail_count(width: f32, wave: &Waveform, clip: &AudioClip) -> usize {
    let visible = count(width);
    if !wave.duration_seconds.is_finite() || wave.duration_seconds <= 0. {
        return visible;
    }
    let start = seconds(clip.source_in).clamp(0., wave.duration_seconds);
    let end = clip
        .source_out
        .map(seconds)
        .unwrap_or(start + seconds(clip.duration))
        .clamp(start, wave.duration_seconds);
    let ratio = ((end - start) / wave.duration_seconds).max(0.01);
    ((visible as f64 / ratio).ceil() as usize).clamp(visible, 512)
}
impl ImagineApp {
    pub(super) fn timeline_waveform(
        &mut self,
        ui: &mut egui::Ui,
        project: &VideoProject,
        clip: &AudioClip,
        bar: egui::Rect,
    ) {
        if !bar.intersects(ui.clip_rect()) {
            return;
        }
        let reveal = helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
        #[cfg(feature = "ui-preview")]
        ui.ctx().data_mut(|d| {
            d.insert_temp(
                egui::Id::new(("waveform-painted", &clip.id.0)),
                json!({"visible":false,"hidden":!reveal}),
            )
        });
        if !reveal {
            return;
        }
        let Some(asset) = clip.asset.as_ref().and_then(|id| project.assets.get(id)) else {
            return;
        };
        let state = &mut self.timeline.waveform;
        if state.project.as_ref() != Some(&project.id) {
            *state = State {
                project: Some(project.id.clone()),
                ..Default::default()
            };
        }
        let visible = count(bar.width());
        let best = state
            .values
            .iter()
            .filter(|((id, revision, _), _)| id == &asset.id && *revision == asset.revision)
            .filter_map(|((_, _, n), v)| v.as_ref().ok().map(|v| (*n, v)))
            .max_by_key(|(n, _)| *n);
        let needed = best
            .map(|(_, v)| detail_count(bar.width(), v, clip))
            .unwrap_or(visible);
        if let Some((_, wave)) = best {
            let values = bars(wave, clip, visible);
            let painter = ui.painter().with_clip_rect(bar.intersect(ui.clip_rect()));
            for (i, v) in values.iter().enumerate() {
                let x = bar.left() + (i as f32 + 0.5) / visible as f32 * bar.width();
                let half = (*v as f32).clamp(0., 1.) * (bar.height() - 4.) / 2.;
                painter.vline(
                    x,
                    bar.center().y - half..=bar.center().y + half,
                    // Keep the waveform visible on both the ordinary field
                    // and the selected clip's accent background.
                    egui::Stroke::new(1., helto::TEXT.gamma_multiply(0.75)),
                );
            }
            #[cfg(feature="ui-preview")]
            ui.ctx().data_mut(|d|d.insert_temp(egui::Id::new(("waveform-painted",&clip.id.0)),json!({"visible":true,"hidden":false,"count":values.len(),"maximum":values.iter().copied().fold(0.,f64::max),"range":range(wave,clip),"requested":needed})));
        }
        let key = (asset.id.clone(), asset.revision, needed);
        if state.values.contains_key(&key) || !state.loading.is_empty() {
            return;
        }
        let path = self
            .timeline
            .document
            .as_ref()
            .and_then(|d| d.runtime_asset(&asset.id.0))
            .map(|id| format!("/v1/video/assets/{id}/waveform?peaks={needed}"));
        let (Some(client), Some(session), Some(path)) =
            (self.client.clone(), self.timeline.session.clone(), path)
        else {
            return;
        };
        state.loading.insert(key.clone());
        let epoch = self.timeline.epoch;
        let project = project.id.clone();
        let tx = self.tx.clone();
        let ctx = ui.ctx().clone();
        std::thread::spawn(move || {
            let result = runtime()
                .block_on(client.timeline_request(&session, false, &path, None))
                .map_err(|e| e.message);
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Waveform {
                    epoch,
                    project,
                    asset: key.0,
                    revision: key.1,
                    peaks: key.2,
                    result,
                }),
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn director_source_crop_peak_resampling_volume_and_width() {
        let wave = Waveform {
            duration_seconds: 10.,
            peaks: (0..10).map(|i| f64::from(i) / 10.).collect(),
        };
        let mut clip = AudioClip::new(
            AudioClipId("own_waveform".into()),
            super::super::editing::time(0.),
            super::super::editing::time(1.),
            &AudioSettings::default(),
        );
        clip.source_in = super::super::editing::time(5.);
        clip.source_out = Some(super::super::editing::time(7.5));
        clip.volume = 50.;
        assert_eq!(range(&wave, &clip), (5, 8));
        assert_eq!(bars(&wave, &clip, 4), vec![0.25, 0.3, 0.35, 0.35]);
        assert_eq!(count(1.), 16);
        assert_eq!(count(49.), 25);
        assert_eq!(count(9999.), 512);
        assert_eq!(detail_count(49., &wave, &clip), 100);
        let oracle: Value = serde_json::from_str(include_str!("group8-director.json")).unwrap();
        let expected = oracle["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == "waveform_crop_volume")
            .unwrap();
        assert_eq!(serde_json::json!(bars(&wave, &clip, 4)), expected["bars"]);
        assert_eq!(
            detail_count(49., &wave, &clip),
            expected["requested"].as_u64().unwrap() as usize
        );
        clip.volume = 200.;
        assert_eq!(bars(&wave, &clip, 4), vec![1.; 4]);
        clip.volume = 0.;
        assert_eq!(bars(&wave, &clip, 4), vec![0.; 4]);
    }
}
