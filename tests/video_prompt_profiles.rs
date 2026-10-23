//! Oracle captured from Director's source-owned v3 profiles, never runtime config.
use imagine_server::video::authoring::profiles::{Catalog, OPERATIONS, PROVIDERS, resolve};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/video/prompt-studio/profiles")
}
#[test]
fn all_five_profiles_prompts_revisions_and_sixty_operation_resolutions_match_director() {
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/director-prompt-profiles.json")).unwrap();
    let catalog = Catalog::load(&root()).unwrap();
    assert_eq!(catalog.default_id, "normal");
    assert!(catalog.warnings.is_empty());
    let actual = serde_json::to_value(&catalog.profiles).unwrap();
    for (actual, expected) in actual
        .as_array()
        .unwrap()
        .iter()
        .zip(oracle["profiles"].as_array().unwrap())
    {
        for (key, value) in actual.as_object().unwrap() {
            assert!(
                value == &expected[key],
                "profile {} differs at {key}",
                actual["id"]
            );
        }
    }
    assert_eq!(catalog.profiles.len(), 5);
    assert_eq!(
        serde_json::to_value(&catalog.prompts).unwrap(),
        oracle["prompts"]
    );
    for profile in &catalog.profiles {
        for provider in PROVIDERS {
            for operation in OPERATIONS {
                assert_eq!(
                    json!(resolve(&profile.generation, provider, operation).unwrap()),
                    oracle["resolved"][&profile.id][provider][operation],
                    "{} {provider} {operation}",
                    profile.id
                );
            }
        }
    }
    for file in oracle["source_files"].as_array().unwrap() {
        let bytes = std::fs::read(root().join(file["path"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            file["sha256"].as_str().unwrap()
        );
    }
}
fn copy_profiles(destination: &Path) {
    for entry in std::fs::read_dir(root()).unwrap() {
        let entry = entry.unwrap();
        let folder = destination.join(entry.file_name());
        std::fs::create_dir(&folder).unwrap();
        for file in std::fs::read_dir(entry.path()).unwrap() {
            let file = file.unwrap();
            std::fs::copy(file.path(), folder.join(file.file_name())).unwrap();
        }
    }
}
#[test]
fn legacy_temperature_and_v2_animation_are_resolved_in_memory_without_rewriting_inputs() {
    let temp = tempfile::tempdir().unwrap();
    copy_profiles(temp.path());
    let legacy = temp.path().join("own_legacy");
    std::fs::create_dir(&legacy).unwrap();
    let v1=serde_json::to_vec(&json!({"id":"own_legacy","label":"Own legacy","default":false,"order":99,"temperature":0.6})).unwrap();
    std::fs::write(legacy.join("profile.json"), &v1).unwrap();
    let v2 = temp.path().join("own_v2");
    std::fs::create_dir(&v2).unwrap();
    let bytes=serde_json::to_vec(&json!({"schema_version":2,"id":"own_v2","label":"Own v2","default":false,"order":100,"generation":{"defaults":{"temperature":0.8},"operations":{},"providers":{}}})).unwrap();
    std::fs::write(v2.join("profile.json"), &bytes).unwrap();
    let catalog = Catalog::load(temp.path()).unwrap();
    assert!(catalog.warnings.is_empty(), "{:?}", catalog.warnings);
    for id in ["own_legacy", "own_v2"] {
        let profile = catalog.snapshot(id).unwrap();
        assert!(!profile.fallback);
        for provider in PROVIDERS {
            assert_eq!(
                resolve(&profile.profile.generation, provider, "animate_image").unwrap(),
                resolve(
                    &profile.profile.generation,
                    provider,
                    "story_scene_expansion"
                )
                .unwrap()
            );
        }
    }
    assert_eq!(
        catalog
            .snapshot("own_legacy")
            .unwrap()
            .profile
            .schema_version,
        1
    );
    assert_eq!(
        catalog.snapshot("own_v2").unwrap().profile.schema_version,
        3
    );
    assert_eq!(std::fs::read(legacy.join("profile.json")).unwrap(), v1);
    assert_eq!(std::fs::read(v2.join("profile.json")).unwrap(), bytes);
}
#[test]
fn reload_changes_future_snapshots_and_keeps_inflight_prompts_and_parameters() {
    let temp = tempfile::tempdir().unwrap();
    copy_profiles(temp.path());
    let before = Catalog::load(temp.path()).unwrap();
    let snapshot = before.snapshot(" nsfw_prism ").unwrap();
    assert_eq!(snapshot.resolved_id, "nsfw_prism");
    assert!(!snapshot.fallback);
    let file = temp.path().join("nsfw_prism/animate_image.txt");
    std::fs::write(&file, "New synthetic animation instruction").unwrap();
    let after = Catalog::load(temp.path())
        .unwrap()
        .snapshot("nsfw_prism")
        .unwrap();
    assert_ne!(snapshot.profile.revision, after.profile.revision);
    assert_ne!(
        snapshot.prompts["animate_image"],
        after.prompts["animate_image"]
    );
    assert_eq!(
        json!(snapshot.profile.generation),
        json!(after.profile.generation)
    );
    let fallback = before.snapshot("missing").unwrap();
    assert!(fallback.fallback);
    assert_eq!(fallback.resolved_id, "normal");
    assert_eq!(
        fallback.warning,
        "The selected Prompt Studio profile is unavailable; using Normal."
    );
    assert!(!before.snapshot("").unwrap().fallback);
    assert!(resolve(&snapshot.profile.generation, "other", "animate_image").is_err());
}
#[test]
fn child_defaults_override_inherited_operations_and_reasoning_merges_fieldwise() {
    let temp = tempfile::tempdir().unwrap();
    copy_profiles(temp.path());
    let file = temp.path().join("creative/profile.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    value["generation"] = json!({"defaults":{"temperature":0.8},
        "operations":{"story_generation":{"max_tokens":12000}},
        "providers":{"koboldcpp":{"defaults":{"reasoning":{"effort":"high"}},
            "operations":{"story_scene_expansion":{"reasoning":{"budget_tokens":1234}}}}}});
    std::fs::write(file, serde_json::to_vec(&value).unwrap()).unwrap();
    let catalog = Catalog::load(temp.path()).unwrap();
    assert!(catalog.warnings.is_empty());
    let generation = catalog.snapshot("creative").unwrap().profile.generation;
    let outline = resolve(&generation, "koboldcpp", "story_outline").unwrap();
    assert_eq!(outline["temperature"], 0.8);
    assert_eq!(outline["max_tokens"], 12000);
    let expanded = resolve(&generation, "koboldcpp", "story_scene_expansion").unwrap();
    assert_eq!(
        expanded["reasoning"],
        json!({"include":true,"effort":"high","enable_thinking":true,"budget_tokens":1234})
    );
    let animate = resolve(&generation, "koboldcpp", "animate_image").unwrap();
    // Its explicitly inherited provider animation override wins over child global defaults.
    assert_eq!(animate["temperature"], 0.05);
    // Animation has no implicit expansion fallback in v3.
    assert!(animate["reasoning"].get("budget_tokens").is_none());
}
#[test]
fn invalid_default_and_escaping_prompt_symlink_fail_without_reading_external_text() {
    let temp = tempfile::tempdir().unwrap();
    let profiles = temp.path().join("profiles");
    std::fs::create_dir(&profiles).unwrap();
    copy_profiles(&profiles);
    let external = temp.path().join("own-external.txt");
    std::fs::write(&external, "own synthetic outside-root fixture").unwrap();
    let prompt = profiles.join("normal/animate_image.txt");
    std::fs::remove_file(&prompt).unwrap();
    std::os::unix::fs::symlink(&external, &prompt).unwrap();
    assert!(Catalog::load(&profiles).is_err());
}
