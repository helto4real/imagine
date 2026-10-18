//! The selected Flux.2 file uses sd.VAE's original Diffusers conversion.
use crate::native::Error;
use std::collections::BTreeMap;
use tch::Tensor;
fn enabled<'a>(keys: impl Iterator<Item = &'a String>) -> bool {
    let keys = keys.collect::<Vec<_>>();
    keys.iter()
        .any(|k| k.as_str() == "decoder.up_blocks.0.resnets.0.norm1.weight")
        && !keys
            .iter()
            .any(|k| k.as_str() == "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight")
}
fn key(name: &str) -> String {
    let mut out = name
        .replace("conv_shortcut", "nin_shortcut")
        .replace("conv_norm_out", "norm_out")
        .replace("mid_block.attentions.0.", "mid.attn_1.");
    for i in 0..4 {
        for j in 0..2 {
            out = out.replace(
                &format!("encoder.down_blocks.{i}.resnets.{j}."),
                &format!("encoder.down.{i}.block.{j}."),
            );
        }
        if i < 3 {
            out = out.replace(
                &format!("down_blocks.{i}.downsamplers.0."),
                &format!("down.{i}.downsample."),
            );
            out = out.replace(
                &format!("up_blocks.{i}.upsamplers.0."),
                &format!("up.{}.upsample.", 3 - i),
            );
        }
        for j in 0..3 {
            out = out.replace(
                &format!("decoder.up_blocks.{i}.resnets.{j}."),
                &format!("decoder.up.{}.block.{j}.", 3 - i),
            );
        }
    }
    for i in 0..2 {
        out = out.replace(
            &format!("mid_block.resnets.{i}."),
            &format!("mid.block_{}.", i + 1),
        );
    }
    if name.contains("attentions") {
        for (source, target) in [
            ("group_norm.", "norm."),
            ("query.", "q."),
            ("key.", "k."),
            ("value.", "v."),
            ("to_q.", "q."),
            ("to_k.", "k."),
            ("to_v.", "v."),
            ("to_out.0.", "proj_out."),
            ("proj_attn.", "proj_out."),
        ] {
            out = out.replace(source, target);
        }
    }
    out
}
fn attention_weight(name: &str) -> bool {
    ["q", "k", "v", "proj_out"]
        .iter()
        .any(|part| name.contains(&format!("mid.attn_1.{part}.weight")))
}
pub fn flux2_vae_checkpoint_shapes(state: Vec<(String, Vec<i64>)>) -> BTreeMap<String, Vec<i64>> {
    if !enabled(state.iter().map(|(name, _)| name)) {
        return state.into_iter().collect();
    }
    let conv3d = state
        .iter()
        .any(|(name, shape)| key(name).ends_with(".conv.weight") && shape.len() == 5);
    state
        .into_iter()
        .map(|(name, mut shape)| {
            let name = key(&name);
            if attention_weight(&name) {
                shape.extend(std::iter::repeat_n(1, if conv3d { 3 } else { 2 }));
            }
            (name, shape)
        })
        .collect()
}
pub fn flux2_vae_checkpoint(
    state: Vec<(String, Tensor)>,
) -> Result<BTreeMap<String, Tensor>, Error> {
    if !enabled(state.iter().map(|(name, _)| name)) {
        return Ok(state.into_iter().collect());
    }
    let conv3d = state
        .iter()
        .any(|(name, tensor)| key(name).ends_with(".conv.weight") && tensor.dim() == 5);
    state
        .into_iter()
        .map(|(name, tensor)| {
            let name = key(&name);
            let tensor = if attention_weight(&name) {
                let mut shape = tensor.size();
                shape.extend(std::iter::repeat_n(1, if conv3d { 3 } else { 2 }));
                tensor.f_reshape(shape.as_slice())?
            } else {
                tensor
            };
            Ok((name, tensor))
        })
        .collect()
}
