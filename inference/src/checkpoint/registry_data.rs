//! Generated frozen class match data; not model forward implementations.
//! comfy/supported_models_base.py SHA256 0b234d6f6ceacd04ff96e4126c388cd7c763e21a7d1aed672ba3a235d7a914a5
//! comfy/supported_models.py SHA256 9893e77e3af02e2a71158523a0c6b37d724fd9ecb0f5fe6dcdbf51d6033e0f78
use super::model_registry::ClassDef;
use crate::{execution_value::Value as V, patching::options::Options};
pub(super) fn classes() -> Vec<ClassDef> {
    vec![
        ClassDef {
            name: "BASE",
            parent: None,
            own_config: Some(Options::from_value(V::Dict(vec![]))),
            own_required: Some(vec![]),
        },
        ClassDef {
            name: "SD15",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("context_dim".into(), V::Int(768)),
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(false)),
                ("adm_in_channels".into(), V::None),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SD20",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("context_dim".into(), V::Int(1024)),
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                ("adm_in_channels".into(), V::None),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SD21UnclipL",
            parent: Some(2),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("context_dim".into(), V::Int(1024)),
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                ("adm_in_channels".into(), V::Int(1536)),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SD21UnclipH",
            parent: Some(2),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("context_dim".into(), V::Int(1024)),
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                ("adm_in_channels".into(), V::Int(2048)),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SDXLRefiner",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(384)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                ("context_dim".into(), V::Int(1280)),
                ("adm_in_channels".into(), V::Int(2560)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(0),
                        V::Int(0),
                        V::Int(4),
                        V::Int(4),
                        V::Int(4),
                        V::Int(4),
                        V::Int(0),
                        V::Int(0),
                    ]),
                ),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SDXL",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(0),
                        V::Int(0),
                        V::Int(2),
                        V::Int(2),
                        V::Int(10),
                        V::Int(10),
                    ]),
                ),
                ("context_dim".into(), V::Int(2048)),
                ("adm_in_channels".into(), V::Int(2816)),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SSD1B",
            parent: Some(6),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(0),
                        V::Int(0),
                        V::Int(2),
                        V::Int(2),
                        V::Int(4),
                        V::Int(4),
                    ]),
                ),
                ("context_dim".into(), V::Int(2048)),
                ("adm_in_channels".into(), V::Int(2816)),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Segmind_Vega",
            parent: Some(6),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(0),
                        V::Int(0),
                        V::Int(1),
                        V::Int(1),
                        V::Int(2),
                        V::Int(2),
                    ]),
                ),
                ("context_dim".into(), V::Int(2048)),
                ("adm_in_channels".into(), V::Int(2816)),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "KOALA_700M",
            parent: Some(6),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![V::Int(0), V::Int(2), V::Int(5)]),
                ),
                ("context_dim".into(), V::Int(2048)),
                ("adm_in_channels".into(), V::Int(2816)),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "KOALA_1B",
            parent: Some(6),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![V::Int(0), V::Int(2), V::Int(6)]),
                ),
                ("context_dim".into(), V::Int(2048)),
                ("adm_in_channels".into(), V::Int(2816)),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SVD_img2vid",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("in_channels".into(), V::Int(8)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(0),
                        V::Int(0),
                    ]),
                ),
                ("context_dim".into(), V::Int(1024)),
                ("adm_in_channels".into(), V::Int(768)),
                ("use_temporal_attention".into(), V::Bool(true)),
                ("use_temporal_resblock".into(), V::Bool(true)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SV3D_u",
            parent: Some(11),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("in_channels".into(), V::Int(8)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(0),
                        V::Int(0),
                    ]),
                ),
                ("context_dim".into(), V::Int(1024)),
                ("adm_in_channels".into(), V::Int(256)),
                ("use_temporal_attention".into(), V::Bool(true)),
                ("use_temporal_resblock".into(), V::Bool(true)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SV3D_p",
            parent: Some(12),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("in_channels".into(), V::Int(8)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(1),
                        V::Int(0),
                        V::Int(0),
                    ]),
                ),
                ("context_dim".into(), V::Int(1024)),
                ("adm_in_channels".into(), V::Int(1280)),
                ("use_temporal_attention".into(), V::Bool(true)),
                ("use_temporal_resblock".into(), V::Bool(true)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Stable_Zero123",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("context_dim".into(), V::Int(768)),
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(false)),
                ("adm_in_channels".into(), V::None),
                ("use_temporal_attention".into(), V::Bool(false)),
                ("in_channels".into(), V::Int(8)),
            ]))),
            own_required: Some(vec![
                "cc_projection.weight".into(),
                "cc_projection.bias".into(),
            ]),
        },
        ClassDef {
            name: "SD_X4Upscaler",
            parent: Some(2),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("context_dim".into(), V::Int(1024)),
                ("model_channels".into(), V::Int(256)),
                ("in_channels".into(), V::Int(7)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                ("adm_in_channels".into(), V::None),
                ("use_temporal_attention".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Stable_Cascade_C",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "stable_cascade_stage".into(),
                V::Text("c".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Stable_Cascade_B",
            parent: Some(16),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "stable_cascade_stage".into(),
                V::Text("b".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "SD15_instructpix2pix",
            parent: Some(1),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("context_dim".into(), V::Int(768)),
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(false)),
                ("adm_in_channels".into(), V::None),
                ("use_temporal_attention".into(), V::Bool(false)),
                ("in_channels".into(), V::Int(8)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SDXL_instructpix2pix",
            parent: Some(6),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                (
                    "transformer_depth".into(),
                    V::List(vec![
                        V::Int(0),
                        V::Int(0),
                        V::Int(2),
                        V::Int(2),
                        V::Int(10),
                        V::Int(10),
                    ]),
                ),
                ("context_dim".into(), V::Int(2048)),
                ("adm_in_channels".into(), V::Int(2816)),
                ("use_temporal_attention".into(), V::Bool(false)),
                ("in_channels".into(), V::Int(8)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "LotusD",
            parent: Some(2),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("model_channels".into(), V::Int(320)),
                ("use_linear_in_transformer".into(), V::Bool(true)),
                ("use_temporal_attention".into(), V::Bool(false)),
                ("adm_in_channels".into(), V::Int(4)),
                ("in_channels".into(), V::Int(4)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "SD3",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("in_channels".into(), V::Int(16)),
                ("pos_embed_scaling_factor".into(), V::None),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "StableAudio",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "audio_model".into(),
                V::Text("dit1.0".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "StableAudio3",
            parent: Some(22),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("audio_model".into(), V::Text("dit1.0".into())),
                ("global_cond_shared_embed".into(), V::Bool(true)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "AuraFlow",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "cond_seq_dim".into(),
                V::Int(2048),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "PixArtAlpha",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("pixart_alpha".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "PixArtSigma",
            parent: Some(25),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("pixart_sigma".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanDiT",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("hydit".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanDiT1",
            parent: Some(27),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("hydit1".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Flux",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("flux".into())),
                ("guidance_embed".into(), V::Bool(true)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "FluxInpaint",
            parent: Some(29),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("flux".into())),
                ("guidance_embed".into(), V::Bool(true)),
                ("in_channels".into(), V::Int(96)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "FluxSchnell",
            parent: Some(29),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("flux".into())),
                ("guidance_embed".into(), V::Bool(false)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Flux2",
            parent: Some(29),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("flux2".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Flux3",
            parent: Some(32),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("flux3".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Lens",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("lens".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "GenmoMochi",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("mochi_preview".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "LTXV",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("ltxv".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "LTXAV",
            parent: Some(36),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("ltxav".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "MiniMaxH3",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("minimax_h3".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanVideo",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("hunyuan_video".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanVideoI2V",
            parent: Some(39),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("hunyuan_video".into())),
                ("in_channels".into(), V::Int(33)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanVideoSkyreelsI2V",
            parent: Some(39),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("hunyuan_video".into())),
                ("in_channels".into(), V::Int(32)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "CosmosT2V",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("cosmos".into())),
                ("in_channels".into(), V::Int(16)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "CosmosI2V",
            parent: Some(42),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("cosmos".into())),
                ("in_channels".into(), V::Int(17)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "CosmosT2IPredict2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("cosmos_predict2".into())),
                ("in_channels".into(), V::Int(16)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Anima",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("anima".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "CosmosI2VPredict2",
            parent: Some(44),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("cosmos_predict2".into())),
                ("in_channels".into(), V::Int(17)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Lumina2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("lumina2".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "ZImage",
            parent: Some(47),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("lumina2".into())),
                ("dim".into(), V::Int(3840)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "ZImagePixelSpace",
            parent: Some(48),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("zimage_pixel".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "MingImage",
            parent: Some(48),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("ming_image".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "PixelDiTT2I",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("pixeldit_t2i".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "PiD",
            parent: Some(51),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("pid".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_T2V",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("t2v".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_CausalAR_T2V",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("t2v".into())),
                ("causal_ar".into(), V::Bool(true)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_I2V",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("i2v".into())),
                ("in_dim".into(), V::Int(36)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_FunControl2V",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("i2v".into())),
                ("in_dim".into(), V::Int(48)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_Camera",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("camera".into())),
                ("in_dim".into(), V::Int(32)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN22_Camera",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("camera_2.2".into())),
                ("in_dim".into(), V::Int(36)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_Vace",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("vace".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_HuMo",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("humo".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN22_S2V",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("s2v".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN22_Animate",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("animate".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN_Animate2",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("animate2".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN22_T2V",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("t2v".into())),
                ("out_dim".into(), V::Int(48)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Trellis2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("trellis2".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_FlowRVS",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("flow_rvs".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_SCAIL",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("scail".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN21_SCAIL2",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("scail2".into())),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "WAN22_WanDancer",
            parent: Some(53),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("wan2.1".into())),
                ("model_type".into(), V::Text("wandancer".into())),
                ("in_dim".into(), V::Int(36)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Hunyuan3Dv2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("hunyuan3d2".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Hunyuan3Dv2_1",
            parent: Some(70),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("hunyuan3d2_1".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Hunyuan3Dv2mini",
            parent: Some(70),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("hunyuan3d2".into())),
                ("depth".into(), V::Int(8)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "TripoSplat",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("triposplat".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "HiDream",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("hidream".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "HiDreamO1",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("hidream_o1".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "SenseNovaU15",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("sensenova_u15".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Chroma",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("chroma".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "SeedVR2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("seedvr2".into()),
            )]))),
            own_required: Some(vec![
                "{}positive_conditioning".into(),
                "{}negative_conditioning".into(),
            ]),
        },
        ClassDef {
            name: "ChromaRadiance",
            parent: Some(77),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("chroma_radiance".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "ACEStep",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "audio_model".into(),
                V::Text("ace".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Omnigen2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("omnigen2".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Boogu",
            parent: Some(81),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("boogu".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Ideogram4",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("ideogram4".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Krea2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("krea2".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "MOSSImage",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("moss_image".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "MageFlow",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("mage_flow".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "QwenImage",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("qwen_image".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "QwenImage21",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("qwen_image21".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "JoyImage",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("joyimage".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanImage21",
            parent: Some(39),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("hunyuan_video".into())),
                ("vec_in_dim".into(), V::None),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanImage21Refiner",
            parent: Some(39),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("hunyuan_video".into())),
                (
                    "patch_size".into(),
                    V::List(vec![V::Int(1), V::Int(1), V::Int(1)]),
                ),
                ("vec_in_dim".into(), V::None),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanVideo15",
            parent: Some(39),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("hunyuan_video".into())),
                ("vision_in_dim".into(), V::Int(1152)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "HunyuanVideo15_SR_Distilled",
            parent: Some(39),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("hunyuan_video".into())),
                ("vision_in_dim".into(), V::Int(1152)),
                ("in_channels".into(), V::Int(98)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "Kandinsky5",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("kandinsky5".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "Kandinsky5Image",
            parent: Some(94),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("kandinsky5".into())),
                ("model_dim".into(), V::Int(2560)),
                ("visual_embed_dim".into(), V::Int(64)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "ACEStep15",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "audio_model".into(),
                V::Text("ace1.5".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "YuE2",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "audio_model".into(),
                V::Text("yue2".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "MiniMaxMusic3",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "audio_model".into(),
                V::Text("minimax_music3".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "LongCatImage",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("flux".into())),
                ("guidance_embed".into(), V::Bool(false)),
                ("vec_in_dim".into(), V::None),
                ("context_in_dim".into(), V::Int(3584)),
                ("txt_ids_dims".into(), V::List(vec![V::Int(1), V::Int(2)])),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "RT_DETR_v4",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("RT_DETR_v4".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "DepthAnything3",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("DepthAnything3".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "ErnieImage",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("ernie".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "SAM3",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("SAM3".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "SAM31",
            parent: Some(103),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("SAM31".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "CogVideoX_T2V",
            parent: Some(0),
            own_config: Some(Options::from_value(V::Dict(vec![(
                "image_model".into(),
                V::Text("cogvideox".into()),
            )]))),
            own_required: None,
        },
        ClassDef {
            name: "CogVideoX_I2V",
            parent: Some(105),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("cogvideox".into())),
                ("in_channels".into(), V::Int(32)),
            ]))),
            own_required: None,
        },
        ClassDef {
            name: "CogVideoX_Inpaint",
            parent: Some(105),
            own_config: Some(Options::from_value(V::Dict(vec![
                ("image_model".into(), V::Text("cogvideox".into())),
                ("in_channels".into(), V::Int(48)),
            ]))),
            own_required: None,
        },
    ]
}
pub(super) const ORDER: &[usize] = &[
    20, 14, 18, 1, 2, 3, 4, 19, 5, 6, 7, 9, 10, 8, 15, 16, 17, 12, 13, 21, 23, 22, 24, 25, 26, 27,
    28, 30, 29, 99, 31, 35, 36, 37, 38, 93, 92, 91, 90, 41, 40, 39, 42, 43, 44, 46, 49, 50, 48, 52,
    51, 47, 64, 54, 53, 55, 56, 59, 57, 58, 61, 60, 62, 63, 66, 67, 68, 69, 72, 70, 71, 73, 74, 75,
    76, 77, 78, 79, 80, 96, 98, 97, 81, 82, 86, 87, 88, 89, 83, 84, 85, 32, 33, 34, 95, 94, 45,
    100, 102, 103, 104, 107, 106, 105, 11, 65, 101,
];
