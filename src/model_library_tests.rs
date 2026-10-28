use super::*;
#[test]
fn namespaced_model_catalog_preserves_ui_selection_across_save_and_restart() {
    let root = tempfile::tempdir().unwrap();
    let models = root.path().join("own-models");
    std::fs::create_dir(&models).unwrap();
    let path = root.path().join("home/.config/imagine/models.json");
    let runtime = root.path().join("runtime/imagine");
    let selection = json!({"model_group":"own-alternative-model","model_groups":{"own-alternative-model":{"variant":"own-file.safetensors","loras":[]}}});
    crate::settings_store::write_namespace(&path, "selection", selection.clone(), &runtime)
        .unwrap();
    let mut library =
        Library::load_namespaced(models.clone(), path.clone(), runtime.clone()).unwrap();
    let mut settings = library.settings.clone();
    settings.components.insert(
        "own-model".into(),
        Components {
            text_encoder: "own-other-encoder".into(),
            vae: "own-other-vae".into(),
        },
    );
    library.update(settings.clone()).unwrap();
    assert_eq!(
        crate::settings_store::read_namespace(&path, "selection").unwrap(),
        Some(selection)
    );
    let restored = Library::load_namespaced(models, path, runtime).unwrap();
    assert_eq!(
        serde_json::to_value(restored.settings).unwrap(),
        serde_json::to_value(settings).unwrap()
    );
}
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
        fixture::diffusion(false, false, true),
    );
    library.scan().unwrap();
    let snapshot = library.snapshot();
    let model = snapshot
        .models
        .iter()
        .find(|m| m["model"] == contract::BF16)
        .unwrap();
    assert_eq!(model["group"], QWEN_GROUP);
    assert_eq!(model["name"], BF16_DIFFUSION);
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
    mismatched.validate().unwrap(); // Syntax validation no longer trusts filenames.
    assert_eq!(
        library.pin_explicit(&mismatched).unwrap_err().code,
        "model_settings_mismatch"
    );
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
        fixture::diffusion(true, true, false),
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
    assert_eq!(meta["constraints"]["reference_images"]["max_count"], 4);
    assert_eq!(meta["constraints"]["control_maps"]["depth"], true);
    let (mut value,_)=library.prepare(json!({"model":contract::MOSS,"operation":"text_to_image","prompt":"synthetic","model_selection":{"diffusion":diffusion},"sampling":{"steps":4,"cfg_scale":3.,"sampler":"euler","scheduler":"normal"}})).unwrap();
    value["width"] = json!(256);
    value["height"] = json!(256);
    let mut request: contract::JobRequest = serde_json::from_value(value).unwrap();
    request.validate().unwrap();
    request.inputs.reference_images.push("synthetic".into());
    assert_eq!(request.validate(), Err("unexpected_reference_images"));
    // MOSS edits through an edit LoRA with up to four references.
    request.operation = contract::Operation::ImageEdit;
    request.validate().unwrap();
    request.inputs.reference_images = vec!["synthetic".into(); 5];
    assert_eq!(request.validate(), Err("unsupported_moss_configuration"));
    request.inputs.reference_images.truncate(1);
    // Edits add the edit LoRA once; control maps steer text to image only.
    assert_eq!(request.missing_moss_adapters(), [contract::MOSS_EDIT_LORA]);
    request.add_moss_adapters();
    assert!(request.missing_moss_adapters().is_empty());
    request.inputs.depth_map = Some("depth".into());
    assert_eq!(request.validate(), Err("unsupported_control_maps"));
    request.inputs.reference_images.clear();
    request.operation = contract::Operation::TextToImage;
    request
        .model_options
        .moss_image
        .as_mut()
        .unwrap()
        .loras
        .clear();
    assert_eq!(
        request.missing_moss_adapters(),
        [contract::MOSS_CONTROL_LORA]
    );
    request.add_moss_adapters();
    request.validate().unwrap();
    assert_eq!(
        request.asset_ids().collect::<Vec<_>>(),
        [&"depth".to_string()]
    );
    // Region prompts name colors of the color map, each once, never black.
    let region = |color: &str, prompt: &str| contract::RegionPrompt {
        color: color.into(),
        prompt: prompt.into(),
        character: false,
    };
    request.inputs.regions = vec![region("#e6194b", "a person")];
    assert_eq!(request.validate(), Err("invalid_region_prompts"));
    request.inputs.color_map = Some("regions".into());
    request.add_moss_adapters();
    request.validate().unwrap();
    for bad in [
        vec![region("#000000", "black")],
        vec![region("e6194b", "no hash")],
        vec![region("#e6194b", " ")],
        vec![region("#e6194b", "a"), region("#E6194B", "same")],
    ] {
        request.inputs.regions = bad;
        assert_eq!(request.validate(), Err("invalid_region_prompts"));
    }
    request.inputs.regions.clear();
    request.inputs.color_map = None;
    request.model_options.moss_control = Some(contract::ControlOptions {
        strength: 1.,
        start: 0.6,
        end: 0.2,
    });
    assert_eq!(request.validate(), Err("invalid_control_options"));
    request.inputs.depth_map = None;
    assert_eq!(request.validate(), Err("unexpected_control_options"));
    request.model_options.moss_control = None;
    // Control maps come from one photo, without prompt, references or maps.
    let mut maps = request.clone();
    maps.operation = contract::Operation::ControlMaps;
    maps.prompt.clear();
    maps.inputs.source_image = Some("photo".into());
    maps.validate().unwrap();
    assert_eq!(maps.asset_ids().collect::<Vec<_>>(), [&"photo".to_string()]);
    assert_eq!(crate::model_adapter::output_count(&maps), 2);
    maps.inputs.depth_map = Some("depth".into());
    assert_eq!(maps.validate(), Err("unsupported_control_maps"));
    maps.inputs.depth_map = None;
    maps.inputs.source_image = None;
    assert_eq!(maps.validate(), Err("unsupported_control_maps"));
    maps.inputs.source_image = Some("photo".into());
    maps.operation = contract::Operation::TextToImage;
    assert_eq!(maps.validate(), Err("unsupported_control_maps"));
    // Library selections resolve the adapters as verified library files.
    assert_eq!(
        library
            .prepare(
                json!({"model":contract::MOSS,"operation":"text_to_image","prompt":"synthetic",
                "inputs":{"color_map":"color"},"model_selection":{"diffusion":diffusion},
                "sampling":{"steps":4,"cfg_scale":3.,"sampler":"euler","scheduler":"normal"}})
            )
            .unwrap_err()
            .code,
        "moss_adapter_missing"
    );
    fixture::write(
        &root,
        &format!("loras/{}", contract::MOSS_CONTROL_LORA),
        json!({"synthetic.lora_A.weight":fixture::tensor(&[1])}),
    );
    library.scan().unwrap();
    let (value, resolved) = library
        .prepare(
            json!({"model":contract::MOSS,"operation":"text_to_image","prompt":"synthetic",
            "inputs":{"color_map":"color"},"model_selection":{"diffusion":diffusion},
            "sampling":{"steps":4,"cfg_scale":3.,"sampler":"euler","scheduler":"normal"}}),
        )
        .unwrap();
    assert_eq!(
        value["model_options"]["moss_image"]["loras"],
        json!([{"path":contract::MOSS_CONTROL_LORA,"strength":1.0}])
    );
    resolved.unwrap().verify().unwrap();
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

#[test]
fn arbitrary_checkpoint_names_are_identified_by_content_and_profiles_persist() {
    let (dir, mut library) = setup();
    let root = dir.path().join("models");
    fixture::write(
        &root,
        "diffusion_models/my-finetune.weights",
        fixture::diffusion(false, false, true),
    );
    fixture::write(
        &root,
        "diffusion_models/moss-merge",
        fixture::diffusion(true, false, true),
    );
    // An official-looking filename must not grant admission to wrong shapes.
    let mut wrong = fixture::diffusion(false, false, true);
    wrong["img_in.weight"]["shape"] = json!([128, 64]);
    fixture::write(&root, &format!("diffusion_models/{BF16_DIFFUSION}"), wrong);
    library.scan().unwrap();
    let qwen = id(&library, "my-finetune.weights");
    let moss = id(&library, "moss-merge");
    assert_eq!(
        library
            .file(&qwen, "diffusion_models")
            .unwrap()
            .model
            .as_deref(),
        Some(contract::BF16)
    );
    assert_eq!(
        library
            .file(&moss, "diffusion_models")
            .unwrap()
            .model
            .as_deref(),
        Some(contract::MOSS_BF16)
    );
    assert!(
        !library
            .files
            .iter()
            .find(|f| f.path == format!("diffusion_models/{BF16_DIFFUSION}"))
            .unwrap()
            .supported
    );
    let mut settings = library.settings.clone();
    settings
        .execution_profiles
        .insert(qwen.clone(), "turbo".into());
    library.update(settings).unwrap();
    assert_eq!(
        library
            .file(&qwen, "diffusion_models")
            .unwrap()
            .model
            .as_deref(),
        Some(contract::TURBO)
    );
    let restored = Library::load(root, dir.path().join("settings/library.json")).unwrap();
    assert_eq!(
        restored
            .file(&qwen, "diffusion_models")
            .unwrap()
            .model
            .as_deref(),
        Some(contract::TURBO)
    );
    let (value, _) = restored.prepare(json!({"model":contract::TURBO,"operation":"text_to_image","model_selection":{"diffusion":qwen},"sampling":{"steps":17,"cfg_scale":2.5,"sampler":"euler","scheduler":"simple"}})).unwrap();
    assert_eq!(value["sampling"]["steps"], 17);
    let (moss_value, _) = restored.prepare(json!({"model":contract::MOSS_BF16,
        "operation":"text_to_image","prompt":"Synthetic BF16",
        "model_selection":{"diffusion":moss,"loras":[{"id":id(&restored,"style.safetensors"),"strength":0.5}]},
        "sampling":{"steps":7,"cfg_scale":3.,"sampler":"euler","scheduler":"normal"}})).unwrap();
    let moss_request: contract::JobRequest = serde_json::from_value(moss_value).unwrap();
    moss_request.validate().unwrap();
    assert_eq!(
        moss_request.model_options.moss_image.unwrap().loras[0].strength,
        0.5
    );
    let metadata = contract::model(contract::MOSS_BF16).unwrap();
    assert_eq!(metadata["defaults"]["model"], contract::MOSS_BF16);
}

#[test]
fn header_detection_distinguishes_layout_precision_and_unsupported_architectures() {
    use crate::model_spec::*;
    for (int8, format) in [
        (true, WeightFormat::Int8ConvRot),
        (false, WeightFormat::Bf16),
    ] {
        for (separate, qkv) in [(true, QkvLayout::Separate), (false, QkvLayout::Fused)] {
            let mut header = fixture::diffusion(true, int8, separate);
            header["aux.weight"] = fixture::tensor(&[1]);
            assert_eq!(
                detect(&header).unwrap(),
                CheckpointSpec {
                    architecture: Architecture::MossImage,
                    format,
                    qkv
                }
            );
            header["layers.0.proj.q.weight"] = fixture::tensor(&[1]);
            assert_eq!(detect(&header), Err("unsupported_qkv_layout"));
        }
    }
    assert_eq!(detect(&json!({})), Err("unsupported_model_architecture"));
    let mut bad = fixture::diffusion(false, false, true);
    bad["img_in.weight"]["dtype"] = json!("F8_E4M3");
    assert_eq!(detect(&bad), Err("unsupported_weight_format"));
    bad = fixture::diffusion(false, false, true);
    bad["transformer_blocks.32.attn.to_q.weight"] = fixture::tensor(&[4096, 4096]);
    assert_eq!(detect(&bad), Err("unsupported_model_configuration"));
}

#[test]
fn legacy_turbo_filename_does_not_restrict_other_checkpoint_contents() {
    let (dir, mut library) = setup();
    let path = format!("diffusion_models/renamed/{TURBO_DIFFUSION}");
    fixture::write(
        &dir.path().join("models"),
        &path,
        fixture::diffusion(true, false, false),
    );
    library.scan().unwrap();
    let file = library.files.iter().find(|f| f.path == path).unwrap();
    assert!(file.supported);
    assert_eq!(file.model.as_deref(), Some(contract::MOSS_BF16));
}
