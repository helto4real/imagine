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
            json!({"txt_in.text_norm.weight":tensor(&[4096]),"img_in.weight":tensor(&[4096,64])}),
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
