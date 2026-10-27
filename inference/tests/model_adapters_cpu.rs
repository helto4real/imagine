//! Small synthetic CPU contracts. No Runtime, native library or model files.
use imagine_inference::{
    checkpoint::PreparedParameters,
    execution_value::Value,
    models::{moss_image as m, qwen_image21 as q},
    patching as p,
};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};
fn lora(name: &str) -> p::LoraState {
    BTreeMap::from([
        (
            format!("{name}.lora_up.weight"),
            Value::tensor(Tensor::ones([2, 1], (Kind::Float, Device::Cpu))),
        ),
        (
            format!("{name}.lora_down.weight"),
            Value::tensor(Tensor::ones([1, 3], (Kind::Float, Device::Cpu))),
        ),
    ])
}
#[test]
fn moss_separate_and_fused_aliases_follow_original_projection_order() {
    let map = m::moss_key_map(&[(
        "diffusion_model.layers.0.proj.qkv.weight".into(),
        vec![6, 3],
    )]);
    let mut source = BTreeMap::new();
    for name in ["q", "k", "v"] {
        source.extend(lora(&format!("diffusion_model.layers.0.proj.{name}")));
    }
    let state = PreparedParameters::plain(BTreeMap::from([(
        "layers.0.proj.qkv.weight".into(),
        Value::tensor(Tensor::zeros([6, 3], (Kind::BFloat16, Device::Cpu))),
    )]));
    let owner =
        p::bind_component("diffusion_model.", &state, &[(source, 1., 1.)], &map, false).unwrap();
    assert_eq!(owner.report.files[0].parsed_targets.len(), 3);
    assert_eq!(
        owner.report.files[0].accepted_targets,
        owner.report.files[0].parsed_targets
    );
    assert_eq!(
        owner.report.files[0].accepted_targets[2].offset,
        Some([0, 4, 2])
    );
    for (i, name) in ["q", "k", "v"].iter().enumerate() {
        for key in [
            format!("diffusion_model.layers.0.proj.{name}"),
            format!("lora_unet_layers_0_proj_{name}"),
        ] {
            let target = &map.iter().find(|(k, _)| k == &key).unwrap().1;
            assert_eq!(target.key, "diffusion_model.layers.0.proj.qkv.weight");
            assert_eq!(target.offset, Some([0, i as i64 * 2, 2]));
        }
    }
    let map = m::moss_key_map(&[("diffusion_model.layers.0.proj.q.weight".into(), vec![2, 3])]);
    assert!(map.iter().all(|(_, t)| t.offset.is_none()));
}
#[test]
fn reports_distinguish_consumed_parsed_and_accepted_across_components() {
    let mut source = lora("diffusion_model.unit");
    source.extend(lora("text_encoder.unit"));
    source.extend(lora("unknown.unit"));
    let selection = vec![(source, 0.5, 0.5)];
    let mapping = ["diffusion_model.unit", "text_encoder.unit"]
        .iter()
        .map(|s| {
            (
                s.to_string(),
                p::Target {
                    key: format!("{s}.weight"),
                    offset: None,
                },
            )
        })
        .collect::<Vec<_>>();
    let state = PreparedParameters::plain(BTreeMap::from([(
        "unit.weight".into(),
        Value::tensor(Tensor::zeros([2, 3], (Kind::Float, Device::Cpu))),
    )]));
    let a = p::bind_component("diffusion_model.", &state, &selection, &mapping, false).unwrap();
    let b = p::bind_component("text_encoder.", &state, &selection, &mapping, true).unwrap();
    let mut report = a.report.clone();
    report.merge(&b.report);
    assert_eq!(report.files[0].consumed.len(), 4);
    assert_eq!(report.files[0].unmatched.len(), 2);
    assert_eq!(report.files[0].parsed_targets.len(), 2);
    assert_eq!(report.files[0].accepted_targets.len(), 2);
    let empty = p::bind_component("diffusion_model.", &state, &selection, &[], false).unwrap();
    assert!(empty.patches.is_empty());
    assert_eq!(empty.report.files[0].unmatched.len(), 6);
}
#[test]
fn qwen_binding_preserves_order_strengths_and_text_only_acceptance() {
    let schema = q::constructor_schema().unwrap();
    let spec = &schema.models["text"].parameters[0];
    let key = format!(
        "text_encoders.qwen3vl_8b.transformer.{}",
        spec.name.trim_end_matches(".weight")
    );
    let state = PreparedParameters::plain(BTreeMap::from([(
        spec.name.clone(),
        Value::tensor(Tensor::zeros([2, 3], (Kind::Float, Device::Cpu))),
    )]));
    let selections = vec![(lora(&key), 1., -0.25), (lora(&key), 0., 0.)];
    let owner = q::component_patch_owner("text", &state, &selections).unwrap();
    assert_eq!(owner.patches[&spec.name].len(), 2);
    assert_eq!(owner.patches[&spec.name][0].strength, -0.25);
    assert_eq!(owner.patches[&spec.name][1].strength, 0.);
    let dit = q::component_patch_owner("dit", &state, &selections).unwrap();
    assert!(dit.patches.is_empty());
    let mut report = owner.report;
    report.merge(&dit.report);
    assert!(
        report
            .files
            .iter()
            .all(|r| r.unmatched.is_empty() && r.accepted_targets.len() == 1)
    );
}
#[test]
fn moss_layout_changes_only_constructor_metadata_and_preserves_total_weight_size() {
    let fused = m::moss_constructor_schema().unwrap();
    let separate = m::moss_constructor_schema_for_layout(false).unwrap();
    let elements = |s: &imagine_inference::components::ConstructorSchema| {
        s.models["dit"]
            .parameters
            .iter()
            .map(|p| p.shape.iter().product::<i64>())
            .sum::<i64>()
    };
    assert_eq!(elements(&fused), elements(&separate));
    assert_eq!(
        separate.models["dit"].parameters.len(),
        fused.models["dit"].parameters.len() + 84
    );
    let a = m::moss_component_inventory("dit").unwrap();
    let b = m::moss_component_inventory_for_layout("dit", false).unwrap();
    assert_eq!(a.model_size, b.model_size);
    assert_eq!(
        a.modules.iter().map(|m| m.module_bytes).sum::<u64>(),
        b.modules.iter().map(|m| m.module_bytes).sum::<u64>()
    );
    assert_eq!(b.modules.len(), a.modules.len() + 84);
}
#[test]
fn moss_aliases_match_comfy_426f5eea_in_order() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/reference-20261027/moss-lora-keys.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let weights = case["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| {
                let shape = w[1].as_array().unwrap().iter();
                (
                    w[0].as_str().unwrap().to_owned(),
                    shape.map(|d| d.as_i64().unwrap()).collect(),
                )
            })
            .collect::<Vec<_>>();
        // The original builds a dict: a repeated alias keeps its first position and
        // takes the last target.
        let mut aliases: Vec<(String, serde_json::Value)> = Vec::new();
        for (alias, target) in m::moss_key_map(&weights) {
            let target = match target.offset {
                None => serde_json::json!(target.key),
                Some(offset) => serde_json::json!([target.key, offset]),
            };
            match aliases.iter_mut().find(|(a, _)| *a == alias) {
                Some(existing) => existing.1 = target,
                None => aliases.push((alias, target)),
            }
        }
        let expected = case["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| (a[0].as_str().unwrap().to_owned(), a[1].clone()))
            .collect::<Vec<_>>();
        assert_eq!(aliases, expected, "fused {}", case["fused"]);
    }
}
