use crate::native::Error;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub struct DetectedConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub num_layers: i64,
    pub attention_head_dim: i64,
    pub num_attention_heads: i64,
    pub context_in_dim: i64,
    pub mlp_ratio: i64,
    pub fused_mlp: bool,
}
/// QwenImage21-specific detection branch; inputs are logical tensor shapes.
/// Full family priority and quantization-metadata loading are separate boundaries.
pub fn detect(
    shapes: &BTreeMap<String, Vec<i64>>,
    prefix: &str,
) -> Result<Option<DetectedConfig>, Error> {
    let key = |name: &str| format!("{prefix}{name}");
    let required = [
        "txt_in.text_norm.weight",
        "modulation.1.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "img_in.weight",
        "proj_out.weight",
    ];
    let fused = key("transformer_blocks.0.img_mlp.gate_up.weight");
    let split = key("transformer_blocks.0.img_mlp.proj.weight");
    if !required.iter().all(|n| shapes.contains_key(&key(n)))
        || !(shapes.contains_key(&fused) || shapes.contains_key(&split))
    {
        return Ok(None);
    }
    let dimension = |name: &str, index: usize| -> Result<i64, Error> {
        shapes[name]
            .get(index)
            .copied()
            .ok_or_else(|| Error::ReferenceException {
                class: "IndexError",
                message: "tuple index out of range".into(),
            })
    };
    let hd = dimension(&key("transformer_blocks.0.attn.norm_q.weight"), 0)?;
    let inner = dimension(&key("img_in.weight"), 0)?;
    if hd == 0 || inner == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "integer division or modulo by zero".into(),
        });
    }
    let mut layers = 0;
    while shapes
        .keys()
        .any(|n| n.starts_with(&format!("{prefix}transformer_blocks.{layers}.")))
    {
        layers += 1;
    }
    let fused_mlp = shapes.contains_key(&fused);
    let width = dimension(if fused_mlp { &fused } else { &split }, 0)?;
    Ok(Some(DetectedConfig {
        in_channels: dimension(&key("img_in.weight"), 1)?,
        out_channels: dimension(&key("proj_out.weight"), 0)?,
        num_layers: layers,
        attention_head_dim: hd,
        num_attention_heads: inner.div_euclid(hd),
        context_in_dim: dimension(&key("txt_in.text_norm.weight"), 0)?,
        mlp_ratio: if fused_mlp {
            width.div_euclid(2).div_euclid(inner)
        } else {
            width.div_euclid(inner)
        },
        fused_mlp,
    }))
}
