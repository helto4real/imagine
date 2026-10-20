//! P01 source/synthetic contract tests. No model, worker, private files or server.
use imagine_server::video::{contract::*, editing::*, planning::*, project::*};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn time(n: u64) -> Time {
    Time { num: n, den: 1 }
}
fn model(adapter: &str) -> ModelSelection {
    ModelSelection {
        adapter: adapter.into(),
        model: format!("synthetic-{adapter}"),
        variant: "reference".into(),
        profile: "test".into(),
        profile_revision: "1".into(),
    }
}
fn request() -> CreateProject {
    CreateProject {
        schema_version: 1,
        default_model: model("h3-test"),
        delivery: DeliverySettings {
            fps: Time {
                num: 30_000,
                den: 1001,
            },
            width: 1280,
            height: 720,
        },
    }
}
fn h3() -> Value {
    json!({"stage_one":{"steps":20,"sampler":"synthetic","scheduler":"synthetic"},"stage_two":null})
}
fn slot(adapter: &str) -> SettingsSlot {
    SettingsSlot {
        model: model(adapter),
        schema_version: 1,
        values: if adapter == "h3-test" {
            h3()
        } else {
            json!({"strength":7})
        },
    }
}
fn scene(id: &str) -> Scene {
    Scene {
        id: SceneId(id.into()),
        start: time(0),
        duration: time(5),
        description: "SYNTHETIC".into(),
        model: ModelChoice::ProjectDefault,
        settings: vec![slot("h3-test")],
        sections: vec![Section {
            id: SectionId(format!("{id}-section")),
            start: time(0),
            duration: time(5),
            prompt: "SYNTHETIC".into(),
        }],
        bindings: vec![],
        continuity: None,
        accepted_take: None,
    }
}
fn edit(p: &mut VideoProject, commands: Vec<Edit>) -> Result<(), Issue> {
    apply(
        p,
        EditRequest {
            schema_version: 1,
            expected_project_revision: p.project_revision,
            commands,
        },
    )
}
fn project() -> VideoProject {
    let mut p = VideoProject::create(request()).unwrap();
    edit(
        &mut p,
        vec![Edit::AddScene {
            scene: Box::new(scene("a")),
        }],
    )
    .unwrap();
    p
}

struct SyntheticAdapter {
    id: &'static str,
    available: bool,
}
impl VideoModelAdapter for SyntheticAdapter {
    fn id(&self) -> &str {
        self.id
    }
    fn capabilities(&self, selection: &ModelSelection) -> Option<Capabilities> {
        if selection != &model(self.id) {
            return None;
        }
        let mut roles = BTreeMap::from([(InputRole::FirstFrame, [MediaKind::Image].into())]);
        if self.id == "h3-test" {
            roles.insert(InputRole::VoiceReference, [MediaKind::Audio].into());
        }
        Some(Capabilities {
            revision: "synthetic-1".into(),
            runtime_available: self.available,
            experimental: true,
            validation_status: ValidationStatus::SyntheticVerified,
            input_roles: roles,
            continuity: if self.id == "h3-test" {
                [
                    ContinuityMethod::MotionContext,
                    ContinuityMethod::MotionContextExp,
                ]
                .into()
            } else {
                [].into()
            },
            incompatible_roles: vec![(InputRole::FirstFrame, InputRole::VoiceReference)],
        })
    }
    fn validate_settings(&self, _: &ModelSelection, settings: &SettingsSlot) -> Vec<Issue> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Other {
            strength: u32,
        }
        if settings.schema_version != 1 {
            return vec![Issue::new("unsupported_settings_schema", "settings")];
        }
        let valid = if self.id == "h3-test" {
            serde_json::from_value::<H3Settings>(settings.values.clone())
                .ok()
                .is_some_and(|s| s.validate().is_ok())
        } else {
            serde_json::from_value::<Other>(settings.values.clone())
                .ok()
                .is_some_and(|s| s.strength <= 10)
        };
        if valid {
            vec![]
        } else {
            vec![Issue::new("invalid_settings", "settings")]
        }
    }
}
fn registry(available: bool) -> AdapterRegistry {
    let mut r = AdapterRegistry::default();
    for id in ["h3-test", "other-test"] {
        r.register(Box::new(SyntheticAdapter { id, available }))
            .unwrap();
    }
    r
}
fn codes(p: &VideoProject, r: &AdapterRegistry) -> Vec<String> {
    validate_scene(p, &SceneId("a".into()), r)
        .unwrap()
        .blockers
        .into_iter()
        .map(|i| i.code)
        .collect()
}
fn voice(p: &mut VideoProject) {
    edit(
        p,
        vec![
            Edit::AddAsset {
                asset: Asset {
                    id: AssetId("voice".into()),
                    revision: 1,
                    kind: MediaKind::Audio,
                },
            },
            Edit::SetBindings {
                scene_id: SceneId("a".into()),
                bindings: vec![Binding {
                    id: BindingId("voice-binding".into()),
                    asset: AssetId("voice".into()),
                    role: InputRole::VoiceReference,
                    enabled: true,
                }],
            },
        ],
    )
    .unwrap();
}

#[test]
fn create_read_edit_is_session_owned_and_revision_checked() {
    let mut service = ProjectService::default();
    let p = service.create(request()).unwrap();
    let changes = EditRequest {
        schema_version: 1,
        expected_project_revision: 0,
        commands: vec![Edit::AddScene {
            scene: Box::new(scene("a")),
        }],
    };
    let updated = service.edit(&p.id, changes.clone()).unwrap();
    assert_eq!(updated.project_revision, 1);
    assert_eq!(
        service.edit(&p.id, changes).unwrap_err().code,
        "revision_conflict"
    );
    assert_eq!(service.read(&p.id).unwrap(), updated);
    let mut detached = service.read(&p.id).unwrap();
    detached.global_prompt = "DETACHED".into();
    assert_ne!(service.read(&p.id).unwrap(), detached);
    assert!(ProjectService::default().read(&p.id).is_err());
}
#[test]
fn batch_failure_restores_every_field_and_revision() {
    let mut p = project();
    let before = p.clone();
    let err = edit(
        &mut p,
        vec![
            Edit::SetGlobalPrompt {
                prompt: "CHANGED".into(),
            },
            Edit::AddScene {
                scene: Box::new(scene("a")),
            },
        ],
    )
    .unwrap_err();
    assert_eq!(err.code, "duplicate_or_invalid_id");
    assert_eq!(p, before);
    assert!(
        edit(
            &mut p,
            vec![Edit::SetSceneDuration {
                scene_id: SceneId("a".into()),
                duration: time(1)
            }]
        )
        .is_err()
    );
    assert_eq!(p, before);
}
#[test]
fn model_switch_preserves_authoring_settings_and_accepted_take() {
    let mut p = project();
    voice(&mut p);
    p.scenes[0].accepted_take = Some(AcceptedTake {
        id: TakeId("take".into()),
        model: model("h3-test"),
        source_fingerprint: "synthetic-source".into(),
    });
    let before = p.clone();
    let fingerprint = input_revision(&p, &SceneId("a".into())).unwrap();
    edit(
        &mut p,
        vec![
            Edit::SetModelSettings {
                scene_id: SceneId("a".into()),
                settings: slot("other-test"),
            },
            Edit::SetSceneModel {
                scene_id: SceneId("a".into()),
                model: ModelChoice::Override(model("other-test")),
            },
        ],
    )
    .unwrap();
    assert!(codes(&p, &registry(true)).contains(&"input_role_unsupported".into()));
    assert_ne!(
        fingerprint,
        input_revision(&p, &SceneId("a".into())).unwrap()
    );
    edit(
        &mut p,
        vec![Edit::SetSceneModel {
            scene_id: SceneId("a".into()),
            model: ModelChoice::ProjectDefault,
        }],
    )
    .unwrap();
    assert_eq!(
        fingerprint,
        input_revision(&p, &SceneId("a".into())).unwrap()
    );
    assert_eq!(p.scenes[0].accepted_take, before.scenes[0].accepted_take);
    assert_eq!(p.scenes[0].sections, before.scenes[0].sections);
    assert_eq!(p.scenes[0].bindings, before.scenes[0].bindings);
    assert!(codes(&p, &registry(true)).is_empty());
}
#[test]
fn changing_project_default_only_changes_inheriting_scenes() {
    let mut p = project();
    let mut b = scene("b");
    b.model = ModelChoice::Override(model("h3-test"));
    edit(&mut p, vec![Edit::AddScene { scene: Box::new(b) }]).unwrap();
    let before = input_revision(&p, &SceneId("b".into())).unwrap();
    edit(
        &mut p,
        vec![Edit::SetDefaultModel {
            model: model("other-test"),
        }],
    )
    .unwrap();
    assert_eq!(p.effective_model(&p.scenes[0]), &model("other-test"));
    assert_eq!(before, input_revision(&p, &SceneId("b".into())).unwrap());
}
#[test]
fn fingerprint_tracks_only_effective_scene_inputs() {
    let mut p = project();
    let id = SceneId("a".into());
    let before = input_revision(&p, &id).unwrap();
    edit(
        &mut p,
        vec![
            Edit::AddScene {
                scene: Box::new(scene("b")),
            },
            Edit::SetModelSettings {
                scene_id: id.clone(),
                settings: slot("other-test"),
            },
        ],
    )
    .unwrap();
    assert_eq!(before, input_revision(&p, &id).unwrap());
    p.scenes[0].accepted_take = Some(AcceptedTake {
        id: TakeId("take".into()),
        model: model("h3-test"),
        source_fingerprint: "source".into(),
    });
    assert_eq!(before, input_revision(&p, &id).unwrap());
    let changed = Section {
        prompt: "CHANGED".into(),
        ..p.scenes[0].sections[0].clone()
    };
    edit(
        &mut p,
        vec![Edit::UpdateSection {
            scene_id: id.clone(),
            section: changed,
        }],
    )
    .unwrap();
    assert_ne!(before, input_revision(&p, &id).unwrap());
}
#[test]
fn unknown_adapter_and_schema_roundtrip_without_fallback() {
    let mut p = project();
    let mut unknown = slot("future");
    unknown.schema_version = 99;
    unknown.values = json!({"future":{"values":[1,2,3]}});
    edit(
        &mut p,
        vec![
            Edit::SetModelSettings {
                scene_id: SceneId("a".into()),
                settings: unknown,
            },
            Edit::SetSceneModel {
                scene_id: SceneId("a".into()),
                model: ModelChoice::Override(model("future")),
            },
        ],
    )
    .unwrap();
    let decoded: VideoProject = serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
    assert_eq!(decoded, p);
    assert_eq!(codes(&p, &registry(true)), vec!["adapter_missing"]);
    p.scenes[0].model = ModelChoice::ProjectDefault;
    p.scenes[0].settings[0].schema_version = 99;
    assert!(codes(&p, &registry(true)).contains(&"unsupported_settings_schema".into()));
}
#[test]
fn support_and_runtime_availability_are_independent() {
    let p = project();
    assert!(codes(&p, &registry(true)).is_empty());
    assert_eq!(codes(&p, &registry(false)), vec!["runtime_unavailable"]);
    let mut r = registry(true);
    assert!(
        r.register(Box::new(SyntheticAdapter {
            id: "h3-test",
            available: true
        }))
        .is_err()
    );
}
#[test]
fn role_type_and_combination_constraints_block_without_deleting_inputs() {
    let mut p = project();
    voice(&mut p);
    edit(
        &mut p,
        vec![Edit::AddAsset {
            asset: Asset {
                id: AssetId("image".into()),
                revision: 1,
                kind: MediaKind::Image,
            },
        }],
    )
    .unwrap();
    p.scenes[0].bindings.push(Binding {
        id: BindingId("frame".into()),
        asset: AssetId("image".into()),
        role: InputRole::FirstFrame,
        enabled: true,
    });
    assert!(codes(&p, &registry(true)).contains(&"input_combination_unsupported".into()));
    p.scenes[0].bindings[0].enabled = false;
    assert!(codes(&p, &registry(true)).is_empty());
    p.assets.get_mut(&AssetId("image".into())).unwrap().kind = MediaKind::Audio;
    assert!(codes(&p, &registry(true)).contains(&"input_role_unsupported".into()));
    assert_eq!(p.scenes[0].bindings.len(), 2);
}
#[test]
fn continuity_is_source_bound_and_latent_compatibility_fails_closed() {
    let mut p = project();
    let mut b = scene("b");
    b.continuity = Some(Continuity {
        source_scene: SceneId("a".into()),
        method: ContinuityMethod::MotionContextExp,
    });
    edit(&mut p, vec![Edit::AddScene { scene: Box::new(b) }]).unwrap();
    let before = input_revision(&p, &SceneId("b".into())).unwrap();
    let report = validate_scene(&p, &SceneId("b".into()), &registry(true)).unwrap();
    assert!(
        report
            .blockers
            .iter()
            .any(|i| i.code == "continuity_source_missing")
    );
    p.scenes[0].accepted_take = Some(AcceptedTake {
        id: TakeId("take".into()),
        model: model("h3-test"),
        source_fingerprint: "source".into(),
    });
    assert_ne!(before, input_revision(&p, &SceneId("b".into())).unwrap());
    assert!(
        validate_scene(&p, &SceneId("b".into()), &registry(true))
            .unwrap()
            .blockers
            .iter()
            .any(|i| i.code == "continuity_artifact_unverified")
    );
    let before = p.clone();
    assert!(
        edit(
            &mut p,
            vec![Edit::RemoveScene {
                scene_id: SceneId("a".into())
            }]
        )
        .is_err()
    );
    assert_eq!(before, p);
}
#[test]
fn duplicate_allocates_child_ids_and_cannot_forge_take() {
    let mut p = project();
    voice(&mut p);
    edit(
        &mut p,
        vec![Edit::DuplicateScene {
            scene_id: SceneId("a".into()),
            new_scene_id: SceneId("copy".into()),
            start: time(5),
        }],
    )
    .unwrap();
    assert_ne!(p.scenes[0].sections[0].id, p.scenes[1].sections[0].id);
    assert_ne!(p.scenes[0].bindings[0].id, p.scenes[1].bindings[0].id);
    assert_eq!(p.scenes[0].bindings[0].asset, p.scenes[1].bindings[0].asset);
    let mut forged = scene("forged");
    forged.accepted_take = Some(AcceptedTake {
        id: TakeId("fake".into()),
        model: model("h3-test"),
        source_fingerprint: "fake".into(),
    });
    assert_eq!(
        edit(
            &mut p,
            vec![Edit::AddScene {
                scene: Box::new(forged)
            }]
        )
        .unwrap_err()
        .code,
        "unexpected_take"
    );
}
#[test]
fn rational_timing_is_exact_and_extreme_wire_values_do_not_overflow() {
    assert!(Time { num: 1, den: 3 }.fits(Time { num: 2, den: 3 }, time(1)));
    assert!(!Time { num: 1, den: 3 }.fits(
        Time { num: 2, den: 3 },
        Time {
            num: 999,
            den: 1000
        }
    ));
    assert!(Time { num: 2, den: 4 }.equivalent(Time { num: 1, den: 2 }));
    let tiny = Time {
        num: 1,
        den: MAX_INTEGER,
    };
    assert!(tiny.fits(tiny, time(1)));
    let huge = Time {
        num: MAX_INTEGER,
        den: 1,
    };
    assert!(!huge.fits(tiny, huge));
    let mut p = project();
    p.scenes[0].sections[0].start.den = 0;
    assert!(p.validate().is_err());
    p = project();
    p.project_revision = MAX_INTEGER;
    let before = p.clone();
    assert!(edit(&mut p, vec![Edit::SetGlobalPrompt { prompt: "x".into() }]).is_err());
    assert_eq!(p, before);
}
#[test]
fn request_versions_and_unknown_fields_are_rejected() {
    let mut r = request();
    r.schema_version = 2;
    assert_eq!(
        VideoProject::create(r).unwrap_err().code,
        "unsupported_schema"
    );
    let mut value = serde_json::to_value(request()).unwrap();
    value["unexpected"] = json!(true);
    assert!(serde_json::from_value::<CreateProject>(value).is_err());
    assert!(
        serde_json::from_value::<Edit>(
            json!({"operation":"set_global_prompt","prompt":"x","unexpected":true})
        )
        .is_err()
    );
    let mut v = h3();
    v["cfg_scale"] = json!(2.5);
    assert!(serde_json::from_value::<H3Settings>(v).is_err());
    assert!(serde_json::from_value::<H3Settings>(json!({})).is_err());
}
#[test]
fn h3_schema_contains_required_handoff_and_no_image_defaults() {
    let mut value = h3();
    value["stage_two"] = json!({"sampling":{"steps":10,"sampler":"test","scheduler":"test"},"model":model("h3-test"),"handoff":"re_noise_video_prediction","trained_latent_upscale_model":"synthetic-upscaler"});
    let settings: H3Settings = serde_json::from_value(value).unwrap();
    settings.validate().unwrap();
    assert_eq!(
        settings.stage_two.unwrap().handoff,
        H3Handoff::ReNoiseVideoPrediction
    );
    let schemas = schemas();
    let text = schemas["h3_settings"].to_string();
    assert!(text.contains("continue_latent"));
    assert!(text.contains("re_noise_video_prediction"));
    assert!(!text.contains("cfg_scale"));
    assert_eq!(schemas["create_project"]["additionalProperties"], false);
}
#[test]
fn section_edits_roundtrip_and_apply_atomically() {
    fn roundtrip<T: Serialize + for<'de> Deserialize<'de>>(v: &T) {
        let _: T = serde_json::from_value(serde_json::to_value(v).unwrap()).unwrap();
    }
    let mut p = project();
    let id = SceneId("a".into());
    let section = Section {
        id: SectionId("extra".into()),
        start: time(1),
        duration: time(1),
        prompt: "EXTRA".into(),
    };
    let commands = vec![
        Edit::MoveScene {
            scene_id: id.clone(),
            start: time(1),
        },
        Edit::SetDelivery {
            delivery: p.delivery.clone(),
        },
        Edit::SetSceneDescription {
            scene_id: id.clone(),
            description: "NEW".into(),
        },
        Edit::AddSection {
            scene_id: id.clone(),
            section: section.clone(),
        },
        Edit::UpdateSection {
            scene_id: id.clone(),
            section: Section {
                prompt: "UPDATED".into(),
                ..section
            },
        },
        Edit::RemoveSection {
            scene_id: id.clone(),
            section_id: SectionId("extra".into()),
        },
        Edit::SetContinuity {
            scene_id: id,
            continuity: None,
        },
    ];
    for c in &commands {
        roundtrip(c);
    }
    edit(&mut p, commands).unwrap();
    assert_eq!(p.scenes[0].sections.len(), 1);
    assert_eq!(p.scenes[0].start, time(1));
}

#[test]
fn canonical_fingerprint_ignores_json_key_order_and_equivalent_times() {
    let mut p = project();
    let before = input_revision(&p, &SceneId("a".into())).unwrap();
    p.scenes[0].settings[0].values=serde_json::from_str(r#"{"stage_two":null,"stage_one":{"scheduler":"synthetic","sampler":"synthetic","steps":20}}"#).unwrap();
    p.scenes[0].duration = Time { num: 10, den: 2 };
    p.scenes[0].sections[0].duration = Time { num: 15, den: 3 };
    assert_eq!(before, input_revision(&p, &SceneId("a".into())).unwrap());
}
#[test]
fn rational_comparison_matches_exact_small_integer_oracle() {
    for a in 0..8 {
        for b in 1..8 {
            for c in 0..8 {
                for d in 1..8 {
                    for e in 0..8 {
                        for f in 1..8 {
                            let actual = Time { num: a, den: b }
                                .fits(Time { num: c, den: d }, Time { num: e, den: f });
                            assert_eq!(actual, (a * d + c * b) * f <= e * b * d);
                        }
                    }
                }
            }
        }
    }
    assert!(!time(0).fits(Time { num: 1, den: 0 }, time(5)));
}
#[test]
fn checked_in_schemas_and_examples_follow_rust_types() {
    let checked: Value =
        serde_json::from_str(include_str!("../design/video-contract/schemas.json")).unwrap();
    assert_eq!(schemas(), checked);
    let example: CreateProject =
        serde_json::from_str(include_str!("../design/video-contract/create-project.json")).unwrap();
    VideoProject::create(example).unwrap();
    let example: EditRequest =
        serde_json::from_str(include_str!("../design/video-contract/edit-project.json")).unwrap();
    let mut p = VideoProject::create(request()).unwrap();
    apply(&mut p, example).unwrap();
    assert_eq!(p.scenes.len(), 1);
}
