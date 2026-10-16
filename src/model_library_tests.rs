use super::*;
#[path = "../shared/model_catalog_fixture.rs"]
mod fixture;
fn setup() -> (tempfile::TempDir, Library) {
    let dir = tempfile::tempdir().unwrap();
    fixture::models(&dir.path().join("models"));
    let library = Library::load(
        dir.path().join("models"),
        dir.path().join("settings/library.json"),
    )
    .unwrap();
    (dir, library)
}
fn id(library: &Library, path: &str) -> String {
    library
        .files
        .iter()
        .find(|f| f.path.ends_with(path))
        .unwrap()
        .id
        .clone()
}
fn request(library: &Library, loras: Value) -> Value {
    json!({"model":contract::MODEL,"operation":"text_to_image","prompt":"Synthetic test","model_selection":{"diffusion":id(library,DIFFUSION),"loras":loras}})
}
#[test]
fn discovery_uses_recursive_comfy_paths_aliases_and_bounded_headers() {
    let (dir, mut library) = setup();
    let root = dir.path().join("models");
    std::fs::rename(root.join("text_encoders"), root.join("clip")).unwrap();
    std::fs::rename(root.join("diffusion_models"), root.join("unet")).unwrap();
    std::os::unix::fs::symlink(root.join("loras"), root.join("loras/loop")).unwrap();
    std::os::unix::fs::symlink(
        root.join("loras/nested/style.safetensors"),
        root.join("loras/duplicate.safetensors"),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("checkpoints")).unwrap();
    std::fs::write(root.join("checkpoints/unknown.gguf"), b"not loaded").unwrap();
    std::fs::write(
        root.join("loras/broken.safetensors"),
        u64::MAX.to_le_bytes(),
    )
    .unwrap();
    library.scan().unwrap();
    let snapshot = library.snapshot();
    assert_eq!(snapshot.models.len(), 2);
    assert!(snapshot.models.iter().all(|m| m["available"] == true));
    assert_eq!(
        snapshot.files.iter().filter(|f| f.kind == "loras").count(),
        4
    );
    assert!(
        !snapshot
            .files
            .iter()
            .find(|f| f.path.ends_with("unknown.gguf"))
            .unwrap()
            .supported
    );
    assert!(
        snapshot
            .files
            .iter()
            .find(|f| f.path.ends_with("broken.safetensors"))
            .unwrap()
            .supported
    );
    assert!(snapshot.warnings.is_empty());
    let (value, _) = library.prepare(request(&library, json!([]))).unwrap();
    assert!(
        value["model_options"]["qwen_image_2_1"]["generation"]["diffusion_model"]
            .as_str()
            .unwrap()
            .starts_with("unet/")
    );
    let typed: contract::JobRequest = serde_json::from_value(value).unwrap();
    typed.validate().unwrap();
}
#[test]
fn nested_loras_are_selectable_without_header_or_metadata_checks() {
    let (dir, mut library) = setup();
    let root = dir.path().join("models");
    for extension in [
        "safetensors",
        "SAFETENSORS",
        "safetensor",
        "gguf",
        "ckpt",
        "pt",
        "pth",
        "bin",
        "sft",
    ] {
        let relative = format!("loras/styles/portraits/deep/uninspected.{extension}");
        let path = root.join(&relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Deliberately not a weight header: discovery must never inspect its contents.
        std::fs::write(&path, b"synthetic uninspected contents").unwrap();
    }
    std::fs::write(
        root.join("loras/styles/portraits/deep/readme.txt"),
        b"sidecar",
    )
    .unwrap();
    library.scan().unwrap();
    let snapshot = library.snapshot();
    assert!(snapshot.warnings.is_empty());
    assert!(
        !snapshot
            .files
            .iter()
            .any(|f| f.path.ends_with("readme.txt"))
    );
    let loras: Vec<_> = snapshot
        .files
        .iter()
        .filter(|f| f.kind == "loras")
        .collect();
    assert_eq!(loras.len(), 12);
    for file in loras {
        assert!(file.supported);
        assert!(file.group.is_none());
        let (value, resolved) = library
            .prepare(request(&library, json!([{"id":file.id,"strength":0.7}])))
            .unwrap();
        assert_eq!(
            value["model_options"]["qwen_image_2_1"]["generation"]["loras"][0]["path"],
            file.path.strip_prefix("loras/").unwrap()
        );
        resolved.unwrap().verify().unwrap();
    }
}
#[test]
fn legacy_lora_assignments_do_not_restrict_selection_after_restart() {
    let (dir, mut library) = setup();
    let unknown = id(&library, "unknown.safetensors");
    let other = id(&library, "other.safetensors");
    let mut settings = library.settings.clone();
    settings
        .lora_groups
        .insert(unknown.clone(), QWEN_GROUP.into());
    settings.lora_groups.insert(other.clone(), "other".into());
    library.update(settings).unwrap();
    let restarted =
        Library::load(dir.path().join("unused"), library.settings_path.clone()).unwrap();
    let (value, _) = restarted
        .prepare(request(
            &restarted,
            json!([
                {"id":unknown,"strength":0.6},{"id":other,"strength":-0.5}
            ]),
        ))
        .unwrap();
    let loras = &value["model_options"]["qwen_image_2_1"]["generation"]["loras"];
    assert_eq!(loras[0]["strength"], json!(0.6f32));
    assert_eq!(loras[1]["strength"], json!(-0.5f32));
    assert_eq!(
        std::fs::metadata(&library.settings_path).unwrap().mode() & 0o777,
        0o600
    );
}
#[test]
fn selection_is_frozen_across_root_changes_and_rejects_replaced_or_missing_files() {
    let (dir, mut library) = setup();
    let lora = id(&library, "style.safetensors");
    let (value, resolved) = library
        .prepare(request(&library, json!([{"id":lora,"strength":-0.5}])))
        .unwrap();
    let resolved = resolved.unwrap();
    resolved.verify().unwrap();
    let next = dir.path().join("other-models");
    fixture::models(&next);
    let mut settings = library.settings.clone();
    settings.root = next.clone();
    library.update(settings).unwrap();
    assert_ne!(resolved.root, next);
    resolved.verify().unwrap();
    assert_eq!(
        value["model_options"]["qwen_image_2_1"]["generation"]["loras"][0]["path"],
        "nested/style.safetensors"
    );
    let (empty, _) = library.prepare(request(&library, json!([]))).unwrap();
    assert_eq!(
        empty["model_options"]["qwen_image_2_1"]["generation"]["loras"],
        json!([])
    );
    let path = resolved.root.join("loras/nested/style.safetensors");
    std::fs::write(&path, b"replacement").unwrap();
    assert!(resolved.verify().is_err());
    std::fs::remove_file(&path).unwrap();
    assert!(resolved.verify().is_err());
}
#[test]
fn duplicate_components_require_selection_and_bad_weights_are_not_ready() {
    let (dir, mut library) = setup();
    let root = dir.path().join("models");
    let duplicate = root.join(format!("clip/duplicate/{ENCODER}"));
    std::fs::create_dir_all(duplicate.parent().unwrap()).unwrap();
    std::fs::copy(
        root.join(format!("text_encoders/nested/{ENCODER}")),
        &duplicate,
    )
    .unwrap();
    library.scan().unwrap();
    assert!(
        library
            .snapshot()
            .models
            .iter()
            .all(|m| m["problem"] == "choose_model_components")
    );
    let diffusion = id(&library, DIFFUSION);
    let mut settings = library.settings.clone();
    settings.components.insert(
        diffusion.clone(),
        Components {
            text_encoder: id(&library, &format!("duplicate/{ENCODER}")),
            vae: id(&library, VAE),
        },
    );
    library.update(settings).unwrap();
    assert_eq!(
        library
            .snapshot()
            .models
            .iter()
            .find(|m| m["id"] == diffusion)
            .unwrap()["available"],
        true
    );
    std::fs::write(
        root.join(format!("diffusion_models/nested/{DIFFUSION}")),
        b"wrong weights",
    )
    .unwrap();
    library.scan().unwrap();
    assert_eq!(
        library
            .snapshot()
            .models
            .iter()
            .find(|m| m["id"] == diffusion)
            .unwrap()["available"],
        false
    );
}
#[test]
fn invalid_selection_never_injects_paths_profiles_or_internal_metadata() {
    let (_dir, library) = setup();
    let lora = id(&library, "style.safetensors");
    for mut value in [request(&library, json!([])), request(&library, json!([]))] {
        value["resolved_models"] = json!({"root":"/"});
        assert_eq!(
            library.prepare(value).unwrap_err().code,
            "read_only_model_metadata"
        );
    }
    let mut value = request(&library, json!([]));
    value["profile"] = json!("qwen");
    assert_eq!(
        library.prepare(value).unwrap_err().code,
        "model_selection_profile_conflict"
    );
    let mut value = request(&library, json!([]));
    value["model_options"] = json!("invalid");
    assert_eq!(library.prepare(value).unwrap_err().code, "invalid_request");
    let mut value = request(&library, json!([]));
    value["model_selection"]["diffusion"] = json!("../../outside");
    assert_eq!(
        library.prepare(value).unwrap_err().code,
        "model_file_missing"
    );
    assert_eq!(
        library
            .prepare(request(
                &library,
                json!([{"id":lora,"strength":1},{"id":lora,"strength":1}])
            ))
            .unwrap_err()
            .code,
        "invalid_lora_selection"
    );
}
#[test]
fn discovered_turbo_keeps_shipped_sampling_and_operation_limits() {
    let (_dir, library) = setup();
    let metadata = model_metadata(contract::TURBO);
    let mut request = metadata["defaults"].clone();
    request["model"] = json!(contract::TURBO);
    request["operation"] = json!("text_to_image");
    request["prompt"] = json!("Synthetic");
    request["model_selection"] = json!({"diffusion":id(&library,TURBO_DIFFUSION),"loras":[]});
    let (value, _) = library.prepare(request).unwrap();
    let typed: contract::JobRequest = serde_json::from_value(value).unwrap();
    typed.validate().unwrap();
    assert_eq!(typed.sampling.steps, 8);
    assert_eq!(typed.sampling.custom_sigmas.len(), 9);
    assert!(metadata.get("character_sheet").is_none());
}

#[test]
fn changed_lora_files_require_refresh_before_a_new_selection() {
    let (dir, mut library) = setup();
    let lora = id(&library, "style.safetensors");
    std::fs::write(
        dir.path().join("models/loras/nested/style.safetensors"),
        b"changed",
    )
    .unwrap();
    assert_eq!(
        library
            .prepare(request(&library, json!([{"id":lora,"strength":1}])))
            .unwrap_err()
            .code,
        "model_catalog_stale"
    );
    library.scan().unwrap();
    let (_, resolved) = library
        .prepare(request(&library, json!([{"id":lora,"strength":1}])))
        .unwrap();
    resolved.unwrap().verify().unwrap();
}
