use super::*;
impl ImagineApp {
    pub(in crate::app::timeline) fn timeline_audio_inspector(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) -> bool {
        if self.timeline.project_audio {
            ui.strong("Project audio");
            ui.separator();
            let mut a = self
                .timeline
                .audio_settings_draft
                .clone()
                .unwrap_or_else(|| d.project.audio.clone());
            let mut changed = false;
            ui.add_enabled_ui(enabled, |ui| {
                let r = ui.checkbox(&mut a.use_native_audio, "Include generated audio");
                #[cfg(feature = "ui-preview")]
                crate::app::timeline::qa::control(ui, "project-audio-native".into(), &r);
                changed |= r.changed();
                ui.label("Normalization");
                egui::ComboBox::from_id_salt("project-audio-normalization")
                    .selected_text(match a.normalization_mode {
                        NormalizationMode::IntegratedLufs => "Integrated LUFS",
                        NormalizationMode::Peak => "Peak",
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(
                                &mut a.normalization_mode,
                                NormalizationMode::IntegratedLufs,
                                "Integrated LUFS",
                            )
                            .changed();
                        changed |= ui
                            .selectable_value(
                                &mut a.normalization_mode,
                                NormalizationMode::Peak,
                                "Peak",
                            )
                            .changed();
                    });
                for (label, value, step) in [
                    ("Target LUFS", &mut a.target_lufs, 0.5),
                    ("True peak limit", &mut a.true_peak_limit_db, 0.1),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(label);
                        changed |= ui.add(egui::DragValue::new(value).speed(step)).changed();
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Default volume");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut a.default_volume)
                                .range(0.0..=400.)
                                .suffix(" %"),
                        )
                        .changed();
                });
                for (label, value) in [
                    ("Default fade in", &mut a.default_fade_in_seconds),
                    ("Default fade out", &mut a.default_fade_out_seconds),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(label);
                        changed |= ui
                            .add(
                                egui::DragValue::new(value)
                                    .range(0.0..=f64::MAX)
                                    .speed(0.05)
                                    .suffix(" s"),
                            )
                            .changed();
                    });
                }
            });
            self.timeline.audio_settings_draft = Some(a.clone());
            if changed {
                commands.push(Edit::SetAudioSettings { settings: a });
            }
            return true;
        }
        let Some(id) = self.timeline.audio_primary.clone() else {
            return false;
        };
        let Some(original) = clip(&d.project, &id).cloned() else {
            return true;
        };
        let mut c = self
            .timeline
            .audio_draft
            .clone()
            .filter(|c| c.id == id)
            .unwrap_or_else(|| original.clone());
        let mut changed = false;
        let mut duplicated: Option<AudioClip> = None;
        ui.strong("Audio clip");
        ui.separator();
        ui.add_enabled_ui(enabled, |ui| {
            ui.label("Name");
            let r = helto::textarea(
                ui,
                &mut c.name,
                1,
                "Audio name",
                self.prefs.privacy_mode,
                false,
            );
            #[cfg(feature = "ui-preview")]
            crate::app::timeline::qa::control(ui, "audio-name".into(), &r);
            changed |= r.lost_focus();
            let asset_name = c
                .asset
                .as_ref()
                .and_then(|id| {
                    d.project
                        .assets
                        .keys()
                        .filter(|id| d.project.assets[*id].kind == MediaKind::Audio)
                        .position(|a| a == id)
                })
                .map(|i| format!("Audio {}", i + 1))
                .unwrap_or_else(|| "Choose audio…".into());
            ui.label("Source");
            egui::ComboBox::from_id_salt("audio-clip-source")
                .selected_text(asset_name)
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut c.asset, None, "Unassigned")
                        .changed();
                    for (index, asset) in d
                        .project
                        .assets
                        .values()
                        .filter(|a| a.kind == MediaKind::Audio)
                        .enumerate()
                    {
                        changed |= ui
                            .selectable_value(
                                &mut c.asset,
                                Some(asset.id.clone()),
                                format!("Audio {}", index + 1),
                            )
                            .changed();
                    }
                });
            if let Some(runtime) = c
                .asset
                .as_ref()
                .and_then(|id| d.raw["runtime_assets"][&id.0].as_str())
            {
                let response = ui.button("Preview source");
                #[cfg(feature = "ui-preview")]
                crate::app::timeline::qa::control(ui, "audio-preview-source".into(), &response);
                if response.clicked() {
                    self.timeline.selected_take = None;
                    self.timeline_load_media(ui.ctx(), runtime.to_owned());
                }
            }
            ui.horizontal(|ui| {
                ui.label("Volume");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut c.volume)
                            .range(0.0..=400.)
                            .suffix(" %"),
                    )
                    .changed();
            });
            let mut source = seconds(c.source_in);
            ui.horizontal(|ui| {
                ui.label("Source in");
                if ui
                    .add(
                        egui::DragValue::new(&mut source)
                            .range(0.0..=f64::MAX)
                            .speed(0.05)
                            .suffix(" s"),
                    )
                    .changed()
                {
                    c.source_in = time(source);
                    changed = true;
                }
            });
            let mut bounded = c.source_out.is_some();
            if ui.checkbox(&mut bounded, "Source out").changed() {
                c.source_out =
                    bounded.then(|| c.source_in.checked_add(c.duration).unwrap_or(c.source_in));
                changed = true;
            }
            if let Some(value) = c.source_out {
                let mut value = seconds(value);
                if ui
                    .add(
                        egui::DragValue::new(&mut value)
                            .range(0.0..=f64::MAX)
                            .speed(0.05)
                            .suffix(" s"),
                    )
                    .changed()
                {
                    c.source_out = Some(time(value));
                    changed = true;
                }
            }
            for (label, value) in [("Fade in", &mut c.fade_in), ("Fade out", &mut c.fade_out)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    changed |= ui
                        .add(
                            egui::DragValue::new(value)
                                .range(0.0..=f64::MAX)
                                .speed(0.05)
                                .suffix(" s"),
                        )
                        .changed();
                });
            }
            ui.separator();
            let r = ui.checkbox(&mut c.usage.generation_reference, "Use for generation");
            #[cfg(feature = "ui-preview")]
            crate::app::timeline::qa::control(ui, "audio-use-generation".into(), &r);
            changed |= r.changed();
            let r = ui.checkbox(&mut c.usage.final_mix, "Include in mix");
            #[cfg(feature = "ui-preview")]
            crate::app::timeline::qa::control(ui, "audio-include-mix".into(), &r);
            changed |= r.changed();
            if c.usage.generation_reference {
                ui.label("Target character");
                egui::ComboBox::from_id_salt("audio-reference-target")
                    .selected_text(
                        c.reference
                            .target_reference_id
                            .as_ref()
                            .map(|id| {
                                d.project
                                    .creative
                                    .references
                                    .iter()
                                    .find(|r| &r.id == id)
                                    .map_or_else(|| id.clone(), |r| format!("@{}", r.alias))
                            })
                            .unwrap_or_else(|| "No character target".into()),
                    )
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(
                                &mut c.reference.target_reference_id,
                                None,
                                "No character target",
                            )
                            .changed();
                        for reference in &d.project.creative.references {
                            changed |= ui
                                .selectable_value(
                                    &mut c.reference.target_reference_id,
                                    Some(reference.id.clone()),
                                    format!("@{}", reference.alias),
                                )
                                .changed();
                        }
                    });
                egui::ComboBox::from_id_salt("audio-purpose")
                    .selected_text(purpose(c.reference.purpose))
                    .show_ui(ui, |ui| {
                        for p in [
                            AudioPurpose::VoiceTimbre,
                            AudioPurpose::DialogueLyrics,
                            AudioPurpose::AmbienceSfx,
                            AudioPurpose::MusicStyle,
                            AudioPurpose::BeatRhythm,
                            AudioPurpose::Continuity,
                            AudioPurpose::Custom,
                        ] {
                            changed |= ui
                                .selectable_value(&mut c.reference.purpose, p, purpose(p))
                                .changed();
                        }
                    });
                egui::ComboBox::from_id_salt("audio-relationship")
                    .selected_text(
                        if c.reference.relationship == AudioRelationship::Reference {
                            "Reference"
                        } else {
                            "Weak reference"
                        },
                    )
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(
                                &mut c.reference.relationship,
                                AudioRelationship::Reference,
                                "Reference",
                            )
                            .changed();
                        changed |= ui
                            .selectable_value(
                                &mut c.reference.relationship,
                                AudioRelationship::WeakReference,
                                "Weak reference",
                            )
                            .changed();
                    });
                ui.label("Description");
                changed |= helto::textarea(
                    ui,
                    &mut c.reference.description,
                    2,
                    "Audio reference",
                    self.prefs.privacy_mode,
                    true,
                )
                .lost_focus();
                changed |= ui
                    .add(egui::Slider::new(&mut c.reference.strength, 0.0..=1.).text("Strength"))
                    .changed();
            }
            changed |= ui.checkbox(&mut c.enabled, "Enabled").changed();
            let r = ui.checkbox(&mut c.locked, "Locked");
            #[cfg(feature = "ui-preview")]
            crate::app::timeline::qa::control(ui, "audio-lock".into(), &r);
            changed |= r.changed();
            let response = ui.button("Duplicate clip");
            #[cfg(feature = "ui-preview")]
            crate::app::timeline::qa::control(ui, "audio-duplicate".into(), &response);
            if response.clicked() {
                let mut copy = c.clone();
                copy.id = AudioClipId(format!("audio_clip_{}", uuid::Uuid::new_v4().simple()));
                copy.start = copy.start.checked_add(copy.duration).unwrap_or(copy.start);
                duplicated = Some(copy.clone());
                commands.push(Edit::AddAudioClip {
                    track_id: d
                        .project
                        .audio_tracks
                        .iter()
                        .find(|t| t.clips.iter().any(|v| v.id == c.id))
                        .unwrap()
                        .id
                        .clone(),
                    clip: Box::new(copy),
                });
            }
            let response = ui.add_enabled(!c.locked, egui::Button::new("Remove clip"));
            #[cfg(feature = "ui-preview")]
            crate::app::timeline::qa::control(ui, "audio-remove".into(), &response);
            if response.clicked() {
                commands.push(Edit::RemoveAudioClip {
                    clip_id: c.id.clone(),
                });
            }
        });
        if changed
            && c != original
            && !commands
                .iter()
                .any(|e| matches!(e,Edit::RemoveAudioClip{clip_id} if clip_id==&c.id))
        {
            commands.push(Edit::UpdateAudioClip {
                clip: Box::new(c.clone()),
            });
        }
        if let Some(copy) = duplicated {
            self.timeline.audio_selected = BTreeSet::from([copy.id.clone()]);
            self.timeline.audio_primary = Some(copy.id.clone());
            self.timeline.audio_draft = Some(copy);
        } else {
            self.timeline.audio_draft = Some(c);
        }
        true
    }
}

fn purpose(p: AudioPurpose) -> &'static str {
    match p {
        AudioPurpose::VoiceTimbre => "Voice timbre",
        AudioPurpose::DialogueLyrics => "Dialogue / lyrics",
        AudioPurpose::AmbienceSfx => "Ambience / SFX",
        AudioPurpose::MusicStyle => "Music style",
        AudioPurpose::BeatRhythm => "Beat / rhythm",
        AudioPurpose::Continuity => "Continuity",
        AudioPurpose::Custom => "Custom",
    }
}
