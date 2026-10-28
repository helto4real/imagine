use super::*;

fn request(prompt: &str, refs: &[(usize, String)]) -> Result<Value> {
    make_edit_request(
        prompt,
        "negative",
        EditSource {
            asset_id: "source",
            width: 512,
            height: 288,
        },
        refs,
        &imagine_server::contract::model(MODEL).unwrap(),
        &GenerationSettings::default(),
    )
}

#[test]
fn edit_mapping_preserves_prompt_source_geometry_and_sparse_reference_numbers() {
    let prompt = "  Match reference 3; keep referens 1.\nExactly this text.  ";
    let value = request(prompt, &[(1, "first".into()), (3, "third".into())]).unwrap();
    assert_eq!(value["operation"], "image_edit");
    assert_eq!(
        value["inputs"]["reference_images"],
        json!(["source", "first", "third"])
    );
    assert_eq!(
        (value["width"].as_u64(), value["height"].as_u64()),
        (Some(512), Some(288))
    );
    assert_eq!(value["negative_prompt"], "negative");
    let effective = value["prompt"].as_str().unwrap();
    assert!(effective.contains("Reference 1 / referens 1 = <image2>"));
    assert!(effective.contains("Reference 3 / referens 3 = <image3>"));
    assert!(effective.ends_with(&format!("Instruction:\n{prompt}")));
    let typed: imagine_server::contract::JobRequest = serde_json::from_value(value).unwrap();
    typed.validate().unwrap();
}

#[test]
fn edit_references_are_optional_and_source_reserves_one_slot() {
    assert_eq!(
        request("Change the lighting", &[]).unwrap()["inputs"]["reference_images"],
        json!(["source"])
    );
    let refs = (1..=9)
        .map(|n| (n, format!("asset_{n}")))
        .collect::<Vec<_>>();
    let typed: imagine_server::contract::JobRequest =
        serde_json::from_value(request("All references", &refs).unwrap()).unwrap();
    typed.validate().unwrap();
    let mut too_many = refs;
    too_many.push((10, "asset_10".into()));
    assert!(
        request("All references", &too_many)
            .unwrap_err()
            .to_string()
            .contains("nine")
    );
}

#[test]
fn edit_rejects_missing_bindings_without_rewriting_or_guessing() {
    let refs = [(3, "third".into())];
    for prompt in [
        "Reference 1",
        "referens 0",
        "REFERENCE 2.",
        "referens 999999999999999999999999999999999999",
    ] {
        assert!(
            request(prompt, &refs)
                .unwrap_err()
                .to_string()
                .contains("missing")
        );
    }
    for prompt in [
        "Reference 3",
        "REFERENS 3",
        "myreference 2",
        "reference 3d",
        "a reference image",
    ] {
        assert!(request(prompt, &refs).is_ok(), "{prompt}");
    }
    assert!(request(" ", &[]).is_err());
    assert!(request("x", &[(1, "a".into()), (1, "b".into())]).is_err());
    assert!(request("x", &[(0, "a".into())]).is_err());
    assert!(request("x", &[(1, "".into())]).is_err());
}

#[test]
fn edit_rejects_unsupported_source_geometry() {
    let meta = imagine_server::contract::model(MODEL).unwrap();
    for (width, height) in [(512, 287), (64, 512), (128, 128), (8192, 4096)] {
        assert!(
            make_edit_request(
                "x",
                "",
                EditSource {
                    asset_id: "source",
                    width,
                    height
                },
                &[],
                &meta,
                &GenerationSettings::default()
            )
            .is_err()
        );
    }
}

#[test]
fn moss_edits_send_the_instruction_as_written_with_the_source_first() {
    let prompt = "change the color of the dress to blue";
    let value = make_edit_request(
        prompt,
        "",
        EditSource {
            asset_id: "source",
            width: 512,
            height: 288,
        },
        &[(1, "first".into())],
        &imagine_server::contract::model(imagine_server::contract::MOSS).unwrap(),
        &GenerationSettings::default(),
    )
    .unwrap();
    assert_eq!(value["prompt"], prompt);
    assert_eq!(value["operation"], "image_edit");
    assert_eq!(
        value["inputs"]["reference_images"],
        json!(["source", "first"])
    );
    assert_eq!(
        (value["width"].as_u64(), value["height"].as_u64()),
        (Some(512), Some(288))
    );
    let typed: imagine_server::contract::JobRequest = serde_json::from_value(value).unwrap();
    typed.validate().unwrap();
}
