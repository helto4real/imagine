//! Own synthetic key aliases and actual adapter loading/PatchOwner registration.
use imagine_inference::{
    checkpoint::PreparedParameters,
    execution_value::Value,
    h3_product::{
        loras,
        protocol::options::{Lora, Options},
    },
};
use std::{collections::BTreeMap, path::PathBuf};
use tch::{Device, Kind, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let own = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("Own new component destination required")?,
    );
    if !own.starts_with(
        "/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01",
    ) || own.exists()
    {
        return Err("Own new encrypted destination required".into());
    }
    std::fs::create_dir(&own)?;
    let state = PreparedParameters::plain(BTreeMap::from([(
        "audio_patch_proj.weight".into(),
        Value::tensor(Tensor::ones([4, 3], (Kind::Float, Device::Cpu))),
    )]));
    let up = Tensor::arange(8, (Kind::Float, Device::Cpu)).reshape([4, 2]) * 0.125;
    let down = Tensor::arange(6, (Kind::Float, Device::Cpu)).reshape([2, 3]) * 0.25;
    let mut cases = vec![];
    for (index, alias) in [
        "audio_patch_proj",
        "diffusion_model.audio_patch_proj",
        "lora_unet_audio_patch_proj",
    ]
    .iter()
    .enumerate()
    {
        let path = own.join(format!("own-compatible-{index}.weights"));
        Tensor::write_safetensors(
            &[
                (format!("{alias}.lora_up.weight"), &up),
                (format!("{alias}.lora_down.weight"), &down),
            ],
            &path,
        )?;
        let rows = vec![
            Lora {
                name: path.to_string_lossy().into_owned(),
                strength: 0.75,
            },
            Lora {
                name: path.to_string_lossy().into_owned(),
                strength: -1.25,
            },
            Lora {
                name: "not-a-file".into(),
                strength: 0.,
            },
        ];
        let patches = loras::owner(&state, &own, &rows)?;
        let actual = &patches.patches["audio_patch_proj.weight"];
        assert_eq!(patches.patches.len(), 1);
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].strength, 0.75);
        assert_eq!(actual[1].strength, -1.25);
        let second = Options {
            loras: rows,
            ..Default::default()
        }
        .stage_two();
        assert!(second.loras.is_empty());
        cases.push(serde_json::json!({"alias":alias,"target":"audio_patch_proj.weight","patch_strengths":[actual[0].strength,actual[1].strength],"zero_strength_missing_file_not_read":true,"clean_second_stage":true}));
    }
    let mapping = loras::key_map([
        "audio_patch_proj.weight".into(),
        "audio_patch_proj.bias".into(),
    ])
    .into_iter()
    .map(|(alias, target)| (alias, format!("diffusion_model.{}", target.key)))
    .collect::<BTreeMap<_, _>>();
    println!(
        "{}",
        serde_json::json!({"passed":true,"cases":cases,"mapping":mapping,"device":"cpu","cuda_initialized":false})
    );
    Ok(())
}
