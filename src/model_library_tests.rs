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
fn bf16_selection_retains_precision_loras_and_backend_admission() {
    let (dir, mut library) = setup();
    fixture::write(
        &dir.path().join("models"),
        &format!("diffusion_models/nested/{BF16_DIFFUSION}"),
        json!({"txt_in.text_norm.weight":fixture::tensor(&[4096]),"img_in.weight":fixture::tensor(&[4096,64])}),
    );
    library.scan().unwrap();
    let snapshot = library.snapshot();
    let model = snapshot
        .models
        .iter()
        .find(|m| m["model"] == contract::BF16)
        .unwrap();
    assert_eq!(model["group"], QWEN_GROUP);
    assert_eq!(model["name"], "Qwen Image 2.1 · BF16");
    assert_eq!(model["available"], true);
    assert_eq!(
        model["defaults"]["sampling"],
        json!(contract::Sampling::default())
    );
    let value = json!({"model":contract::BF16,"operation":"text_to_image","prompt":"Synthetic BF16 control",
        "model_selection":{"diffusion":model["id"],"loras":[{"id":id(&library,"style.safetensors"),"strength":0.65}]}});
    let (resolved, _) = library.prepare(value).unwrap();
    let typed: contract::JobRequest = serde_json::from_value(resolved).unwrap();
    typed.validate().unwrap();
    let workflow = typed.model_options.generation(&typed.model).unwrap();
    assert!(workflow.diffusion_model.ends_with(BF16_DIFFUSION));
    assert_eq!(workflow.loras[0].strength, 0.65);
    assert!(!workflow.zero_negative_conditioning);
    crate::engine::validate_selection("rust", &typed.model, &typed.sampling).unwrap();
    for backend in ["native", "comfy"] {
        assert!(crate::engine::validate_selection(backend, &typed.model, &typed.sampling).is_err());
        assert_eq!(
            crate::comfy_engine::model_metadata(model.clone(), backend)["available"],
            false
        );
    }
    let mut mismatched = typed;
    mismatched.model = contract::MODEL.into();
    assert_eq!(mismatched.validate(), Err("model_settings_mismatch"));
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
    assert_eq!(typed.sampling.cfg_scale, 1.);
    assert_eq!(typed.sampling.sampler, "euler");
    assert_eq!(
        typed.sampling.custom_sigmas,
        vec![
            1., 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568, 0.,
        ]
    );
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

#[test]
fn moss_defaults_keep_its_encoder_and_vae_but_explicit_cross_family_choices_are_allowed() {
    let (dir, mut library) = setup();
    let root = dir.path().join("models");
    fixture::write(
        &root,
        "diffusion_models/moss/moss_image_sft_int8_convrot.safetensors",
        json!({"video_embedder.weight":fixture::tensor(&[1]),"layers.41.proj.qkv.weight":fixture::tensor(&[1])}),
    );
    fixture::write(
        &root,
        "text_encoders/moss/qwen3.5_9b_int8_convrot.safetensors",
        json!({"model.layers.0.linear_attn.in_proj_qkv.weight":fixture::tensor(&[1]),"model.layers.3.self_attn.q_proj.weight":fixture::tensor(&[1])}),
    );
    fixture::write(
        &root,
        "vae/flux.2/flux2-vae.safetensors",
        json!({"bn.running_mean":fixture::tensor(&[1]),"decoder.conv_in.weight":fixture::tensor(&[1])}),
    );
    library.scan().unwrap();
    let diffusion = id(&library, "moss_image_sft_int8_convrot.safetensors");
    let components = library.components(&diffusion).unwrap();
    assert_eq!(
        library
            .file(&components.text_encoder, "text_encoders")
            .unwrap()
            .group
            .as_deref(),
        Some(MOSS_GROUP)
    );
    let meta = library
        .snapshot()
        .models
        .into_iter()
        .find(|m| m["model"] == contract::MOSS)
        .unwrap();
    assert_eq!(meta["group"], MOSS_GROUP);
    assert_eq!(meta["constraints"]["reference_images"]["max_count"], 0);
    let (mut value,_)=library.prepare(json!({"model":contract::MOSS,"operation":"text_to_image","prompt":"synthetic","model_selection":{"diffusion":diffusion},"sampling":{"steps":4,"cfg_scale":3.,"sampler":"euler","scheduler":"normal"}})).unwrap();
    value["width"] = json!(256);
    value["height"] = json!(256);
    let mut request: contract::JobRequest = serde_json::from_value(value).unwrap();
    request.validate().unwrap();
    request.inputs.reference_images.push("synthetic".into());
    assert_eq!(request.validate(), Err("unsupported_moss_configuration"));
    library.settings.components.insert(
        diffusion.clone(),
        Components {
            text_encoder: id(&library, ENCODER),
            vae: id(&library, VAE),
        },
    );
    let (value, _) = library
        .prepare(json!({"model":contract::MOSS,"operation":"text_to_image",
        "prompt":"synthetic","model_selection":{"diffusion":diffusion},
        "sampling":{"steps":4,"cfg_scale":3.,"sampler":"euler","scheduler":"normal"}}))
        .unwrap();
    assert!(
        value["model_options"]["moss_image"]["text_encoder"]
            .as_str()
            .unwrap()
            .ends_with(ENCODER)
    );
    assert!(
        value["model_options"]["moss_image"]["vae"]
            .as_str()
            .unwrap()
            .ends_with(VAE)
    );
}

#[test]
fn encoders_and_vaes_are_uninspected_selectable_and_persist_without_family_restrictions() {
    let (dir, mut library) = setup();
    let root = dir.path().join("models");
    // Even recognized names with invalid headers must remain selectable.
    for relative in [
        format!("text_encoders/nested/{ENCODER}"),
        format!("vae/nested/{VAE}"),
    ] {
        std::fs::write(root.join(relative), b"synthetic uninspected contents").unwrap();
    }
    for relative in [
        "clip/other/deep/Custom-encoder.GGUF",
        "vae/other/deep/Custom-decoder.pt",
    ] {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"synthetic uninspected contents").unwrap();
    }
    library.scan().unwrap();
    assert!(library.snapshot().warnings.is_empty());
    let diffusion = id(&library, DIFFUSION);
    let choice = Components {
        text_encoder: id(&library, "Custom-encoder.GGUF"),
        vae: id(&library, "Custom-decoder.pt"),
    };
    let mut settings = library.settings.clone();
    settings
        .components
        .insert(diffusion.clone(), choice.clone());
    library.update(settings).unwrap();
    let mut restarted =
        Library::load(dir.path().join("unused"), library.settings_path.clone()).unwrap();
    assert_eq!(restarted.components(&diffusion).unwrap(), choice);
    assert!(
        restarted
            .files
            .iter()
            .filter(|f| matches!(f.kind.as_str(), "text_encoders" | "vae"))
            .all(|f| f.supported)
    );
    // Component flags/group metadata cannot prohibit an explicit choice either.
    for file in &mut restarted.files {
        if file.id == choice.text_encoder || file.id == choice.vae {
            file.supported = false;
            file.group = Some("other-family".into());
        }
    }
    let (value, resolved) = restarted.prepare(request(&restarted, json!([]))).unwrap();
    let typed: contract::JobRequest = serde_json::from_value(value).unwrap();
    typed.validate().unwrap();
    let workflow = typed.model_options.generation(&typed.model).unwrap();
    assert_eq!(workflow.text_encoder, "clip/other/deep/Custom-encoder.GGUF");
    assert_eq!(workflow.vae, "vae/other/deep/Custom-decoder.pt");
    resolved.unwrap().verify().unwrap();
    let mut invalid = restarted.settings.clone();
    invalid.components.get_mut(&diffusion).unwrap().vae = choice.text_encoder.clone();
    restarted.update(invalid).unwrap();
    assert_eq!(
        restarted
            .prepare(request(&restarted, json!([])))
            .unwrap_err()
            .code,
        "model_file_missing"
    );
    let mut settings = restarted.settings.clone();
    settings.components.insert(diffusion, choice);
    restarted.update(settings).unwrap();
    std::fs::write(
        root.join("vae/other/deep/Custom-decoder.pt"),
        b"changed file",
    )
    .unwrap();
    assert_eq!(
        restarted
            .prepare(request(&restarted, json!([])))
            .unwrap_err()
            .code,
        "model_catalog_stale"
    );
    std::fs::remove_file(root.join("vae/other/deep/Custom-decoder.pt")).unwrap();
    assert_eq!(
        restarted
            .prepare(request(&restarted, json!([])))
            .unwrap_err()
            .code,
        "model_file_missing"
    );
    restarted.scan().unwrap();
    assert_eq!(
        restarted
            .prepare(request(&restarted, json!([])))
            .unwrap_err()
            .code,
        "model_file_missing"
    );
}
