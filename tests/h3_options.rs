//! Synthetic component transport and path resolution, with no files/models/CUDA.
use imagine_server::h3_protocol::{self as p, options::*};
use std::path::Path;
#[test]
fn selected_paths_and_shifts_roundtrip_without_name_or_extension_allowlists() {
    let options = Options {
        loras: vec![],
        components: Components {
            model: Some("diffusion_models/compatible-experiment.bin".into()),
            text_encoder: Some("/arbitrary/compatible-encoder.dat".into()),
            video_vae: Some("vae/own-alternative.weights".into()),
            audio_vae: Some("/other/components/audio.model".into()),
        },
        shift_video: 9.5,
        shift_audio: 2.25,
    };
    assert!(options.valid());
    assert_eq!(
        Components::path(
            Path::new("/model-root"),
            options.components.model.as_deref(),
            "fallback"
        ),
        Path::new("/model-root/diffusion_models/compatible-experiment.bin")
    );
    assert_eq!(
        Components::path(
            Path::new("/model-root"),
            options.components.text_encoder.as_deref(),
            "fallback"
        ),
        Path::new("/arbitrary/compatible-encoder.dat")
    );
    let request: p::Request = serde_json::from_value(serde_json::json!({
        "protocol":p::VERSION,"kind":"generate_video","id":"own","profile":"h3-av-noref-v2",
        "prompt":"Own synthetic scene","seed":42,"options":options
    }))
    .unwrap();
    assert!(request.valid());
    let decoded: p::Request =
        serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    assert_eq!(decoded.options, options);
    assert_eq!(Options::default().shift_video, 12.);
    assert_eq!(Options::default().shift_audio, 3.);
    assert!(
        !Options {
            shift_video: 0.,
            ..options
        }
        .valid()
    );
}
#[test]
fn stage_two_uses_clean_model_and_keeps_components_and_shifts() {
    let options = Options {
        loras: vec![Lora {
            name: "/own/compatible adapter.weights".into(),
            strength: -1.25,
        }],
        ..Default::default()
    };
    let second = options.stage_two();
    assert!(second.loras.is_empty());
    assert_eq!(options.loras.len(), 1);
    assert_eq!(second.components, options.components);
    assert_eq!(second.shift_video, options.shift_video);
    assert_eq!(second.shift_audio, options.shift_audio);
    let restored: Options = serde_json::from_slice(&serde_json::to_vec(&options).unwrap()).unwrap();
    assert_eq!(restored, options);
}
#[test]
fn full_u64_seed_reaches_wire_exactly() {
    for seed in [0, 9_007_199_254_740_993, u64::MAX] {
        let request:p::Request=serde_json::from_value(serde_json::json!({"protocol":p::VERSION,"kind":"generate_video","id":"own","profile":"h3-av-noref-v2","prompt":"Own fixture","seed":seed})).unwrap();
        assert!(request.valid());
        let copy: p::Request =
            serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
        assert_eq!(copy.seed, seed);
    }
}
