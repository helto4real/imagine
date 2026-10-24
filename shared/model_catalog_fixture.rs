//! Tiny synthetic safetensors headers for tests; no real weights or inference.
use serde_json::{Value, json};
use std::path::Path;
pub fn tensor(shape: &[u64]) -> Value {
    json!({"dtype":"F32","shape":shape,"data_offsets":[0,4]})
}
pub fn write(root: &Path, relative: &str, header: Value) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0u8; 4]);
    std::fs::write(path, bytes).unwrap();
}
pub fn models(root: &Path) {
    for name in [
        "qwen_image_2.1_int8_convrot.safetensors",
        "qwen_image_2.1_turbo_bf16.safetensors",
    ] {
        write(
            root,
            &format!("diffusion_models/nested/{name}"),
            diffusion(false, name.contains("int8"), true),
        );
    }
    write(
        root,
        "text_encoders/nested/qwen3vl_8b_int8_convrot.safetensors",
        json!({"model.embed_tokens.weight":tensor(&[1,4096]),"model.layers.0.self_attn.q_norm.weight":tensor(&[128])}),
    );
    write(
        root,
        "vae/nested/qwen_image_2.1_vae_bf16.safetensors",
        json!({"encoder.conv1.weight":tensor(&[96,4,3,3]),"decoder.conv1.weight":tensor(&[1152,64,3,3])}),
    );
    write(
        root,
        "loras/nested/style.safetensors",
        json!({"__metadata__":{"ss_base_model_version":"qwen-image-2.1"},"transformer.lora_A.weight":tensor(&[1])}),
    );
    write(
        root,
        "loras/unknown.safetensors",
        json!({"transformer.lora_down.weight":tensor(&[1])}),
    );
    write(
        root,
        "loras/other.safetensors",
        json!({"__metadata__":{"ss_base_model_version":"sdxl"},"transformer.lora_down.weight":tensor(&[1])}),
    );
}

/// Complete admission signatures with a tiny synthetic payload (never loaded as model weights).
pub fn diffusion(moss: bool, int8: bool, separate: bool) -> Value {
    let mut h = serde_json::Map::new();
    let mut add = |key: String, shape: &[u64]| {
        let mut value = tensor(shape);
        value["dtype"] = json!("BF16");
        h.insert(key, value);
    };
    if moss {
        add("video_embedder.weight".into(), &[3072, 128]);
        add("shared_block_modulation.weight".into(), &[18432, 3072]);
        for i in 0..42 {
            if separate {
                for p in ["q", "k", "v"] {
                    add(format!("layers.{i}.proj.{p}.weight"), &[3072, 3072]);
                }
            } else {
                add(format!("layers.{i}.proj.qkv.weight"), &[9216, 3072]);
            }
            add(
                format!("layers.{i}.proj.ffn_up_gate.weight"),
                &[16384, 3072],
            );
        }
    } else {
        add("txt_in.text_norm.weight".into(), &[4096]);
        add("img_in.weight".into(), &[4096, 64]);
        for i in 0..32 {
            for p in ["q", "k", "v"] {
                add(
                    format!("transformer_blocks.{i}.attn.to_{p}.weight"),
                    &[4096, 4096],
                );
            }
            add(
                format!("transformer_blocks.{i}.img_mlp.gate_up.weight"),
                &[24576, 4096],
            );
        }
    }
    if int8 {
        let primary = if moss { "video_embedder" } else { "img_in" };
        h.get_mut(&format!("{primary}.weight")).unwrap()["dtype"] = json!("I8");
        let mut metadata = tensor(&[4]);
        metadata["dtype"] = json!("U8");
        h.insert(format!("{primary}.comfy_quant"), metadata);
    }
    Value::Object(h)
}
