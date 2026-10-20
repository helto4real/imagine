// Generated from locked comfy/model_detection.py; edit the generator.
// Source SHA256: b0f615387a1808eb01edf92bc301bb79e02c42112b566cf6826b67c45e7d71ed
use super::detector_values::*;
use crate::{native::Error, patching::options::Options};
pub(super) fn run(mut cx: Context) -> Result<Options, Error> {
    // comfy/model_detection.py:45
    cx.set(
        "state_dict_keys",
        cx.function(
            "list",
            vec![cx.method(&cx.get("state_dict")?, "keys", vec![], vec![])?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:47
    if cx.truth(&{let detector_tmp_6={let detector_tmp_7=cx.method(&text("{}cond_layer_logits"),"format",vec![cx.get("key_prefix")?],vec![])?;let detector_tmp_8=cx.get("state_dict_keys")?;cx.compare("In",&detector_tmp_7,&detector_tmp_8)?};if cx.truth(&detector_tmp_6)?{{let detector_tmp_3={let detector_tmp_4=cx.method(&text("{}latent_conditioners.0.weight"),"format",vec![cx.get("key_prefix")?],vec![])?;let detector_tmp_5=cx.get("state_dict_keys")?;cx.compare("In",&detector_tmp_4,&detector_tmp_5)?};if cx.truth(&detector_tmp_3)?{{let detector_tmp_1=cx.method(&text("{}diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight"),"format",vec![cx.get("key_prefix")?],vec![])?;let detector_tmp_2=cx.get("state_dict_keys")?;cx.compare("In",&detector_tmp_1,&detector_tmp_2)?}}else{detector_tmp_3}}}else{detector_tmp_6}})?{// comfy/model_detection.py:52
return cx.mapping(vec![(text("audio_model"),text("minimax_music3"))]);
}
    // comfy/model_detection.py:54
    if cx.truth(&{
        let detector_tmp_9 = cx.method(
            &text("{}joint_blocks.0.context_block.attn.qkv.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_10 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_9, &detector_tmp_10)?
    })? {
        // comfy/model_detection.py:55
        cx.set("unet_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:56
        cx.set_item(
            &cx.get("unet_config")?,
            &text("in_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}x_embedder.proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:57
        cx.set(
            "patch_size",
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}x_embedder.proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(2_i128)?,
            )?,
        );
        // comfy/model_detection.py:58
        cx.set_item(
            &cx.get("unet_config")?,
            &text("patch_size"),
            cx.get("patch_size")?,
        )?;
        // comfy/model_detection.py:59
        cx.set(
            "final_layer",
            cx.method(
                &text("{}final_layer.linear.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:60
        if cx.truth(&{
            let detector_tmp_11 = cx.get("final_layer")?;
            let detector_tmp_12 = cx.get("state_dict")?;
            cx.compare("In", &detector_tmp_11, &detector_tmp_12)?
        })? {
            // comfy/model_detection.py:61
            cx.set_item(
                &cx.get("unet_config")?,
                &text("out_channels"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(
                        &cx.attribute(
                            &cx.index(&cx.get("state_dict")?, &cx.get("final_layer")?)?,
                            "shape",
                        )?,
                        &integer(0_i128)?,
                    )?,
                    &cx.binary("Mult", &cx.get("patch_size")?, &cx.get("patch_size")?)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:63
        cx.set_item(
            &cx.get("unet_config")?,
            &text("depth"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}x_embedder.proj.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &integer(64_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:64
        cx.set_item(&cx.get("unet_config")?, &text("input_size"), none())?;
        // comfy/model_detection.py:65
        cx.set(
            "y_key",
            cx.method(
                &text("{}y_embedder.mlp.0.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:66
        if cx.truth(&{
            let detector_tmp_13 = cx.get("y_key")?;
            let detector_tmp_14 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_13, &detector_tmp_14)?
        })? {
            // comfy/model_detection.py:67
            cx.set_item(
                &cx.get("unet_config")?,
                &text("adm_in_channels"),
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("y_key")?)?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:69
        cx.set(
            "context_key",
            cx.method(
                &text("{}context_embedder.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:70
        if cx.truth(&{
            let detector_tmp_15 = cx.get("context_key")?;
            let detector_tmp_16 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_15, &detector_tmp_16)?
        })? {
            // comfy/model_detection.py:71
            cx.set(
                "in_features",
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("context_key")?)?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            );
            // comfy/model_detection.py:72
            cx.set(
                "out_features",
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("context_key")?)?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
            );
            // comfy/model_detection.py:73
            cx.set_item(
                &cx.get("unet_config")?,
                &text("context_embedder_config"),
                cx.mapping(vec![
                    (text("target"), text("torch.nn.Linear")),
                    (
                        text("params"),
                        cx.mapping(vec![
                            (text("in_features"), cx.get("in_features")?),
                            (text("out_features"), cx.get("out_features")?),
                        ])?,
                    ),
                ])?,
            )?;
        }
        // comfy/model_detection.py:74
        cx.set(
            "num_patches_key",
            cx.method(
                &text("{}pos_embed"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:75
        if cx.truth(&{
            let detector_tmp_17 = cx.get("num_patches_key")?;
            let detector_tmp_18 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_17, &detector_tmp_18)?
        })? {
            // comfy/model_detection.py:76
            cx.set(
                "num_patches",
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("num_patches_key")?)?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            );
            // comfy/model_detection.py:77
            cx.set_item(
                &cx.get("unet_config")?,
                &text("num_patches"),
                cx.get("num_patches")?,
            )?;
            // comfy/model_detection.py:78
            cx.set_item(
                &cx.get("unet_config")?,
                &text("pos_embed_max_size"),
                cx.function(
                    "round",
                    vec![cx.function("math.sqrt", vec![cx.get("num_patches")?], vec![])?],
                    vec![],
                )?,
            )?;
        }
        // comfy/model_detection.py:80
        cx.set(
            "rms_qk",
            cx.method(
                &text("{}joint_blocks.0.context_block.attn.ln_q.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:81
        if cx.truth(&{
            let detector_tmp_19 = cx.get("rms_qk")?;
            let detector_tmp_20 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_19, &detector_tmp_20)?
        })? {
            // comfy/model_detection.py:82
            cx.set_item(&cx.get("unet_config")?, &text("qk_norm"), text("rms"))?;
        }
        // comfy/model_detection.py:84
        cx.set_item(
            &cx.get("unet_config")?,
            &text("pos_embed_scaling_factor"),
            none(),
        )?;
        // comfy/model_detection.py:85
        cx.set(
            "context_processor",
            cx.method(
                &text("{}context_processor.layers.0.attn.qkv.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:86
        if cx.truth(&{
            let detector_tmp_21 = cx.get("context_processor")?;
            let detector_tmp_22 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_21, &detector_tmp_22)?
        })? {
            // comfy/model_detection.py:87
            cx.set_item(
                &cx.get("unet_config")?,
                &text("context_processor_layers"),
                cx.function(
                    "count_blocks",
                    vec![
                        cx.get("state_dict_keys")?,
                        cx.binary(
                            "Add",
                            &cx.method(
                                &text("{}context_processor.layers."),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                            &text("{}."),
                        )?,
                    ],
                    vec![],
                )?,
            )?;
        }
        // comfy/model_detection.py:88
        cx.set_item(
            &cx.get("unet_config")?,
            &text("x_block_self_attn_layers"),
            list(vec![]),
        )?;
        // comfy/model_detection.py:89
        for detector_item in cx.iter(&cx.get("state_dict_keys")?)? {
            cx.set("key", detector_item); // comfy/model_detection.py:90
            if cx.truth(&{
                let detector_tmp_23 = cx.method(
                    &cx.get("key")?,
                    "startswith",
                    vec![cx.method(
                        &text("{}joint_blocks."),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?],
                    vec![],
                )?;
                if cx.truth(&detector_tmp_23)? {
                    cx.method(
                        &cx.get("key")?,
                        "endswith",
                        vec![text(".x_block.attn2.qkv.weight")],
                        vec![],
                    )?
                } else {
                    detector_tmp_23
                }
            })? {
                // comfy/model_detection.py:91
                cx.set(
                    "layer",
                    cx.slice(
                        &cx.get("key")?,
                        Some(cx.function(
                            "len",
                            vec![cx.method(
                                &text("{}joint_blocks."),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?],
                            vec![],
                        )?),
                        Some(cx.unary(
                            "USub",
                            &cx.function("len", vec![text(".x_block.attn2.qkv.weight")], vec![])?,
                        )?),
                        None,
                    )?,
                );
                // comfy/model_detection.py:92
                cx.method(
                    &cx.index(&cx.get("unet_config")?, &text("x_block_self_attn_layers"))?,
                    "append",
                    vec![cx.function("int", vec![cx.get("layer")?], vec![])?],
                    vec![],
                )?;
            }
        }
        // comfy/model_detection.py:93
        return cx.get("unet_config");
    }
    // comfy/model_detection.py:95
    if cx.truth(&{
        let detector_tmp_24 = cx.method(
            &text("{}clf.1.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_25 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_24, &detector_tmp_25)?
    })? {
        // comfy/model_detection.py:96
        cx.set("unet_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:97
        cx.set(
            "text_mapper_name",
            cx.method(
                &text("{}clip_txt_mapper.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:98
        if cx.truth(&{
            let detector_tmp_26 = cx.get("text_mapper_name")?;
            let detector_tmp_27 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_26, &detector_tmp_27)?
        })? {
            // comfy/model_detection.py:99
            cx.set_item(
                &cx.get("unet_config")?,
                &text("stable_cascade_stage"),
                text("c"),
            )?;
            // comfy/model_detection.py:100
            cx.set(
                "w",
                cx.index(&cx.get("state_dict")?, &cx.get("text_mapper_name")?)?,
            );
            // comfy/model_detection.py:101
            if cx.truth(&{
                let detector_tmp_28 =
                    cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(0_i128)?)?;
                let detector_tmp_29 = integer(1536_i128)?;
                cx.compare("Eq", &detector_tmp_28, &detector_tmp_29)?
            })? {
                // comfy/model_detection.py:102
                cx.set_item(
                    &cx.get("unet_config")?,
                    &text("c_cond"),
                    integer(1536_i128)?,
                )?;
                // comfy/model_detection.py:103
                cx.set_item(
                    &cx.get("unet_config")?,
                    &text("c_hidden"),
                    list(vec![integer(1536_i128)?, integer(1536_i128)?]),
                )?;
                // comfy/model_detection.py:104
                cx.set_item(
                    &cx.get("unet_config")?,
                    &text("nhead"),
                    list(vec![integer(24_i128)?, integer(24_i128)?]),
                )?;
                // comfy/model_detection.py:105
                cx.set_item(
                    &cx.get("unet_config")?,
                    &text("blocks"),
                    list(vec![
                        list(vec![integer(4_i128)?, integer(12_i128)?]),
                        list(vec![integer(12_i128)?, integer(4_i128)?]),
                    ]),
                )?;
            } else {
                // comfy/model_detection.py:106
                if cx.truth(&{
                    let detector_tmp_30 =
                        cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(0_i128)?)?;
                    let detector_tmp_31 = integer(2048_i128)?;
                    cx.compare("Eq", &detector_tmp_30, &detector_tmp_31)?
                })? {
                    // comfy/model_detection.py:107
                    cx.set_item(
                        &cx.get("unet_config")?,
                        &text("c_cond"),
                        integer(2048_i128)?,
                    )?;
                }
            }
        } else {
            // comfy/model_detection.py:108
            if cx.truth(&{
                let detector_tmp_32 = cx.method(
                    &text("{}clip_mapper.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_33 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_32, &detector_tmp_33)?
            })? {
                // comfy/model_detection.py:109
                cx.set_item(
                    &cx.get("unet_config")?,
                    &text("stable_cascade_stage"),
                    text("b"),
                )?;
                // comfy/model_detection.py:110
                cx.set(
                    "w",
                    cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}down_blocks.1.0.channelwise.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                );
                // comfy/model_detection.py:111
                if cx.truth(&{
                    let detector_tmp_34 = cx.index(
                        &cx.attribute(&cx.get("w")?, "shape")?,
                        &cx.unary("USub", &integer(1_i128)?)?,
                    )?;
                    let detector_tmp_35 = integer(640_i128)?;
                    cx.compare("Eq", &detector_tmp_34, &detector_tmp_35)?
                })? {
                    // comfy/model_detection.py:112
                    cx.set_item(
                        &cx.get("unet_config")?,
                        &text("c_hidden"),
                        list(vec![
                            integer(320_i128)?,
                            integer(640_i128)?,
                            integer(1280_i128)?,
                            integer(1280_i128)?,
                        ]),
                    )?;
                    // comfy/model_detection.py:113
                    cx.set_item(
                        &cx.get("unet_config")?,
                        &text("nhead"),
                        list(vec![
                            cx.unary("USub", &integer(1_i128)?)?,
                            cx.unary("USub", &integer(1_i128)?)?,
                            integer(20_i128)?,
                            integer(20_i128)?,
                        ]),
                    )?;
                    // comfy/model_detection.py:114
                    cx.set_item(
                        &cx.get("unet_config")?,
                        &text("blocks"),
                        list(vec![
                            list(vec![
                                integer(2_i128)?,
                                integer(6_i128)?,
                                integer(28_i128)?,
                                integer(6_i128)?,
                            ]),
                            list(vec![
                                integer(6_i128)?,
                                integer(28_i128)?,
                                integer(6_i128)?,
                                integer(2_i128)?,
                            ]),
                        ]),
                    )?;
                    // comfy/model_detection.py:115
                    cx.set_item(
                        &cx.get("unet_config")?,
                        &text("block_repeat"),
                        list(vec![
                            list(vec![
                                integer(1_i128)?,
                                integer(1_i128)?,
                                integer(1_i128)?,
                                integer(1_i128)?,
                            ]),
                            list(vec![
                                integer(3_i128)?,
                                integer(3_i128)?,
                                integer(2_i128)?,
                                integer(2_i128)?,
                            ]),
                        ]),
                    )?;
                } else {
                    // comfy/model_detection.py:116
                    if cx.truth(&{
                        let detector_tmp_36 = cx.index(
                            &cx.attribute(&cx.get("w")?, "shape")?,
                            &cx.unary("USub", &integer(1_i128)?)?,
                        )?;
                        let detector_tmp_37 = integer(576_i128)?;
                        cx.compare("Eq", &detector_tmp_36, &detector_tmp_37)?
                    })? {
                        // comfy/model_detection.py:117
                        cx.set_item(
                            &cx.get("unet_config")?,
                            &text("c_hidden"),
                            list(vec![
                                integer(320_i128)?,
                                integer(576_i128)?,
                                integer(1152_i128)?,
                                integer(1152_i128)?,
                            ]),
                        )?;
                        // comfy/model_detection.py:118
                        cx.set_item(
                            &cx.get("unet_config")?,
                            &text("nhead"),
                            list(vec![
                                cx.unary("USub", &integer(1_i128)?)?,
                                integer(9_i128)?,
                                integer(18_i128)?,
                                integer(18_i128)?,
                            ]),
                        )?;
                        // comfy/model_detection.py:119
                        cx.set_item(
                            &cx.get("unet_config")?,
                            &text("blocks"),
                            list(vec![
                                list(vec![
                                    integer(2_i128)?,
                                    integer(4_i128)?,
                                    integer(14_i128)?,
                                    integer(4_i128)?,
                                ]),
                                list(vec![
                                    integer(4_i128)?,
                                    integer(14_i128)?,
                                    integer(4_i128)?,
                                    integer(2_i128)?,
                                ]),
                            ]),
                        )?;
                        // comfy/model_detection.py:120
                        cx.set_item(
                            &cx.get("unet_config")?,
                            &text("block_repeat"),
                            list(vec![
                                list(vec![
                                    integer(1_i128)?,
                                    integer(1_i128)?,
                                    integer(1_i128)?,
                                    integer(1_i128)?,
                                ]),
                                list(vec![
                                    integer(2_i128)?,
                                    integer(2_i128)?,
                                    integer(2_i128)?,
                                    integer(2_i128)?,
                                ]),
                            ]),
                        )?;
                    }
                }
            }
        }
        // comfy/model_detection.py:121
        return cx.get("unet_config");
    }
    // comfy/model_detection.py:123
    cx.set(
        "shape_key",
        cx.method(
            &text("{}img2shape.t_embedder.mlp.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:124
    cx.set(
        "tex_key",
        cx.method(
            &text("{}shape2txt.t_embedder.mlp.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:125
    if cx.truth(&{
        let detector_tmp_40 = {
            let detector_tmp_41 = cx.get("shape_key")?;
            let detector_tmp_42 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_41, &detector_tmp_42)?
        };
        if cx.truth(&detector_tmp_40)? {
            detector_tmp_40
        } else {
            {
                let detector_tmp_38 = cx.get("tex_key")?;
                let detector_tmp_39 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_38, &detector_tmp_39)?
            }
        }
    })? {
        // comfy/model_detection.py:126
        cx.set("has_shape", {
            let detector_tmp_43 = cx.get("shape_key")?;
            let detector_tmp_44 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_43, &detector_tmp_44)?
        });
        // comfy/model_detection.py:127
        cx.set("has_tex", {
            let detector_tmp_45 = cx.get("tex_key")?;
            let detector_tmp_46 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_45, &detector_tmp_46)?
        });
        // comfy/model_detection.py:128
        cx.set(
            "unet_config",
            cx.mapping(vec![
                (text("image_model"), text("trellis2")),
                (
                    text("resolution"),
                    if cx.truth(&cx.method(
                        &{
                            let detector_tmp_47 = cx.get("metadata")?;
                            if cx.truth(&detector_tmp_47)? {
                                detector_tmp_47
                            } else {
                                cx.mapping(vec![])?
                            }
                        },
                        "get",
                        vec![text("is_512")],
                        vec![],
                    )?)? {
                        integer(32_i128)?
                    } else {
                        integer(64_i128)?
                    },
                ),
                (text("init_txt_model"), cx.get("has_tex")?),
                (text("txt_only"), {
                    let detector_tmp_48 = cx.get("has_tex")?;
                    if cx.truth(&detector_tmp_48)? {
                        cx.unary("Not", &cx.get("has_shape")?)?
                    } else {
                        detector_tmp_48
                    }
                }),
            ])?,
        );
        // comfy/model_detection.py:135
        for detector_item in cx.iter(&tuple(vec![
            tuple(vec![text("img2shape"), text("shape"), integer(2048_i128)?]),
            tuple(vec![
                text("shape2txt"),
                text("texture"),
                integer(2048_i128)?,
            ]),
            tuple(vec![
                text("structure_model"),
                text("structure"),
                integer(1024_i128)?,
            ]),
        ]))? {
            let detector_tmp_49 = detector_item;
            cx.set("sub", cx.index(&detector_tmp_49, &integer(0)?)?);
            cx.set("name", cx.index(&detector_tmp_49, &integer(1)?)?);
            cx.set(
                "proj_in_channels",
                cx.index(&detector_tmp_49, &integer(2)?)?,
            ); // comfy/model_detection.py:138
            cx.set(
                "key",
                cx.method(
                    &text("{}{}.blocks.0.cross_attn.proj_linear.weight"),
                    "format",
                    vec![cx.get("key_prefix")?, cx.get("sub")?],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:139
            if cx.truth(&{
                let detector_tmp_50 = cx.get("key")?;
                let detector_tmp_51 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_50, &detector_tmp_51)?
            })? {
                // comfy/model_detection.py:140
                cx.set_item(
                    &cx.get("unet_config")?,
                    &cx.method(
                        &text("image_attn_mode_{}"),
                        "format",
                        vec![cx.get("name")?],
                        vec![],
                    )?,
                    text("proj"),
                )?;
                // comfy/model_detection.py:141
                cx.set_item(
                    &cx.get("unet_config")?,
                    &cx.method(
                        &text("proj_in_channels_{}"),
                        "format",
                        vec![cx.get("name")?],
                        vec![],
                    )?,
                    cx.get("proj_in_channels")?,
                )?;
            }
        }
        // comfy/model_detection.py:142
        return cx.get("unet_config");
    }
    // comfy/model_detection.py:144
    if cx.truth(&{
        let detector_tmp_52 = cx.method(
            &text("{}transformer.rotary_pos_emb.inv_freq"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_53 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_52, &detector_tmp_53)?
    })? {
        // comfy/model_detection.py:145
        cx.set("unet_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:146
        cx.set_item(
            &cx.get("unet_config")?,
            &text("audio_model"),
            text("dit1.0"),
        )?;
        // comfy/model_detection.py:147
        cx.set_item(
            &cx.get("unet_config")?,
            &text("global_cond_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}to_global_embed.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:148
        cx.set(
            "cond_embed",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}to_cond_embed.0.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:149
        cx.set_item(&cx.get("unet_config")?, &text("project_cond_tokens"), {
            let detector_tmp_54 = cx.index(
                &cx.attribute(&cx.get("cond_embed")?, "shape")?,
                &integer(0_i128)?,
            )?;
            let detector_tmp_55 = cx.index(
                &cx.attribute(&cx.get("cond_embed")?, "shape")?,
                &integer(1_i128)?,
            )?;
            cx.compare("NotEq", &detector_tmp_54, &detector_tmp_55)?
        })?;
        // comfy/model_detection.py:150
        cx.set_item(
            &cx.get("unet_config")?,
            &text("embed_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}to_timestep_embed.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:151
        cx.set(
            "mem_tokens",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![
                    cx.method(
                        &text("{}transformer.memory_tokens"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                    none(),
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:152
        cx.set(
            "to_qkv",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![
                    cx.method(
                        &text("{}transformer.layers.0.self_attn.to_qkv.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                    none(),
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:153
        cx.set("differential", boolean(false));
        // comfy/model_detection.py:154
        if cx.truth(&{
            let detector_tmp_56 = cx.get("to_qkv")?;
            let detector_tmp_57 = none();
            cx.compare("IsNot", &detector_tmp_56, &detector_tmp_57)?
        })? {
            // comfy/model_detection.py:155
            if cx.truth(&{
                let detector_tmp_58 = cx.index(
                    &cx.attribute(&cx.get("to_qkv")?, "shape")?,
                    &integer(0_i128)?,
                )?;
                let detector_tmp_59 = cx.binary(
                    "Mult",
                    &cx.index(
                        &cx.attribute(&cx.get("to_qkv")?, "shape")?,
                        &integer(1_i128)?,
                    )?,
                    &integer(5_i128)?,
                )?;
                cx.compare("Eq", &detector_tmp_58, &detector_tmp_59)?
            })? {
                // comfy/model_detection.py:156
                cx.set("differential", boolean(true));
            }
        }
        // comfy/model_detection.py:157
        if cx.truth(&{
            let detector_tmp_60 = cx.get("mem_tokens")?;
            let detector_tmp_61 = none();
            cx.compare("IsNot", &detector_tmp_60, &detector_tmp_61)?
        })? {
            // comfy/model_detection.py:158
            cx.set_item(
                &cx.get("unet_config")?,
                &text("num_memory_tokens"),
                cx.index(
                    &cx.attribute(&cx.get("mem_tokens")?, "shape")?,
                    &integer(0_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:159
        if cx.truth(&{
            let detector_tmp_62 = cx.method(
                &text("{}transformer.layers.0.self_attn.q_norm.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_63 = cx.get("state_dict")?;
            cx.compare("In", &detector_tmp_62, &detector_tmp_63)?
        })? {
            // comfy/model_detection.py:160
            cx.set_item(
                &cx.get("unet_config")?,
                &text("attn_kwargs"),
                cx.mapping(vec![
                    (text("qk_norm"), text("ln")),
                    (text("feat_scale"), boolean(true)),
                ])?,
            )?;
        }
        // comfy/model_detection.py:161
        cx.set(
            "rms_norm",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![
                    cx.method(
                        &text("{}transformer.layers.0.self_attn.q_norm.gamma"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                    none(),
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:162
        if cx.truth(&{
            let detector_tmp_64 = cx.get("rms_norm")?;
            let detector_tmp_65 = none();
            cx.compare("IsNot", &detector_tmp_64, &detector_tmp_65)?
        })? {
            // comfy/model_detection.py:163
            cx.set_item(
                &cx.get("unet_config")?,
                &text("attn_kwargs"),
                cx.mapping(vec![
                    (text("qk_norm"), text("rms")),
                    (text("differential"), cx.get("differential")?),
                ])?,
            )?;
            // comfy/model_detection.py:164
            cx.set_item(
                &cx.get("unet_config")?,
                &text("norm_type"),
                text("rms_norm"),
            )?;
            // comfy/model_detection.py:165
            cx.set_item(
                &cx.get("unet_config")?,
                &text("num_heads"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(&cx.get("unet_config")?, &text("embed_dim"))?,
                    &cx.index(
                        &cx.attribute(&cx.get("rms_norm")?, "shape")?,
                        &integer(0_i128)?,
                    )?,
                )?,
            )?;
        }
        // comfy/model_detection.py:167
        if cx.truth(&{
            let detector_tmp_66 = cx.method(
                &text("{}timestep_features.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_67 = cx.get("state_dict")?;
            cx.compare("In", &detector_tmp_66, &detector_tmp_67)?
        })? {
            // comfy/model_detection.py:168
            cx.set_item(
                &cx.get("unet_config")?,
                &text("timestep_features_type"),
                text("learned"),
            )?;
        } else {
            // comfy/model_detection.py:170
            cx.set_item(
                &cx.get("unet_config")?,
                &text("timestep_features_type"),
                text("expo"),
            )?;
        }
        // comfy/model_detection.py:172
        cx.set(
            "io_channels",
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}postprocess_conv.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        );
        // comfy/model_detection.py:173
        cx.set_item(
            &cx.get("unet_config")?,
            &text("io_channels"),
            cx.get("io_channels")?,
        )?;
        // comfy/model_detection.py:174
        cx.set_item(
            &cx.get("unet_config")?,
            &text("input_concat_dim"),
            cx.binary(
                "Sub",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}transformer.project_in.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
                &cx.get("io_channels")?,
            )?,
        )?;
        // comfy/model_detection.py:176
        cx.set(
            "local_add_cond",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![
                    cx.method(
                        &text("{}transformer.layers.0.to_local_embed.0.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                    none(),
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:177
        if cx.truth(&{
            let detector_tmp_68 = cx.get("local_add_cond")?;
            let detector_tmp_69 = none();
            cx.compare("IsNot", &detector_tmp_68, &detector_tmp_69)?
        })? {
            // comfy/model_detection.py:178
            cx.set_item(
                &cx.get("unet_config")?,
                &text("local_add_cond_dim"),
                cx.index(
                    &cx.attribute(&cx.get("local_add_cond")?, "shape")?,
                    &integer(1_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:180
        cx.set(
            "global_cond_embed",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![
                    cx.method(
                        &text("{}transformer.global_cond_embedder.0.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                    none(),
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:181
        if cx.truth(&{
            let detector_tmp_70 = cx.get("global_cond_embed")?;
            let detector_tmp_71 = none();
            cx.compare("IsNot", &detector_tmp_70, &detector_tmp_71)?
        })? {
            // comfy/model_detection.py:182
            cx.set_item(
                &cx.get("unet_config")?,
                &text("global_cond_shared_embed"),
                boolean(true),
            )?;
            // comfy/model_detection.py:183
            cx.set_item(
                &cx.get("unet_config")?,
                &text("global_cond_type"),
                text("adaLN"),
            )?;
        }
        // comfy/model_detection.py:185
        cx.set_item(
            &cx.get("unet_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}transformer.layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:186
        return cx.get("unet_config");
    }
    // comfy/model_detection.py:188
    if cx.truth(&{
        let detector_tmp_72 = cx.method(
            &text("{}double_layers.0.attn.w1q.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_73 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_72, &detector_tmp_73)?
    })? {
        // comfy/model_detection.py:189
        cx.set("unet_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:190
        cx.set_item(
            &cx.get("unet_config")?,
            &text("max_seq"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}positional_encoding"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:191
        cx.set_item(
            &cx.get("unet_config")?,
            &text("cond_seq_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}cond_seq_linear.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:192
        cx.set(
            "double_layers",
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}double_layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:193
        cx.set(
            "single_layers",
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}single_layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:194
        cx.set_item(
            &cx.get("unet_config")?,
            &text("n_double_layers"),
            cx.get("double_layers")?,
        )?;
        // comfy/model_detection.py:195
        cx.set_item(
            &cx.get("unet_config")?,
            &text("n_layers"),
            cx.binary("Add", &cx.get("double_layers")?, &cx.get("single_layers")?)?,
        )?;
        // comfy/model_detection.py:196
        return cx.get("unet_config");
    }
    // comfy/model_detection.py:198
    if cx.truth(&{
        let detector_tmp_74 = cx.method(
            &text("{}mlp_t5.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_75 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_74, &detector_tmp_75)?
    })? {
        // comfy/model_detection.py:199
        cx.set("unet_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:200
        cx.set_item(&cx.get("unet_config")?, &text("image_model"), text("hydit"))?;
        // comfy/model_detection.py:201
        cx.set_item(
            &cx.get("unet_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:202
        cx.set_item(
            &cx.get("unet_config")?,
            &text("hidden_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}x_embedder.proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:203
        if cx.truth(&{
            let detector_tmp_78 = {
                let detector_tmp_79 = cx.index(&cx.get("unet_config")?, &text("hidden_size"))?;
                let detector_tmp_80 = integer(1408_i128)?;
                cx.compare("Eq", &detector_tmp_79, &detector_tmp_80)?
            };
            if cx.truth(&detector_tmp_78)? {
                {
                    let detector_tmp_76 = cx.index(&cx.get("unet_config")?, &text("depth"))?;
                    let detector_tmp_77 = integer(40_i128)?;
                    cx.compare("Eq", &detector_tmp_76, &detector_tmp_77)?
                }
            } else {
                detector_tmp_78
            }
        })? {
            // comfy/model_detection.py:204
            cx.set_item(
                &cx.get("unet_config")?,
                &text("mlp_ratio"),
                float(4.3637_f64),
            )?;
        }
        // comfy/model_detection.py:205
        if cx.truth(&{
            let detector_tmp_81 = cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}extra_embedder.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?;
            let detector_tmp_82 = integer(3968_i128)?;
            cx.compare("Eq", &detector_tmp_81, &detector_tmp_82)?
        })? {
            // comfy/model_detection.py:206
            cx.set_item(&cx.get("unet_config")?, &text("size_cond"), boolean(true))?;
            // comfy/model_detection.py:207
            cx.set_item(
                &cx.get("unet_config")?,
                &text("use_style_cond"),
                boolean(true),
            )?;
            // comfy/model_detection.py:208
            cx.set_item(
                &cx.get("unet_config")?,
                &text("image_model"),
                text("hydit1"),
            )?;
        }
        // comfy/model_detection.py:209
        return cx.get("unet_config");
    }
    // comfy/model_detection.py:211
    if cx.truth(&{
        let detector_tmp_83 = cx.method(
            &text("{}txt_in.individual_token_refiner.blocks.0.norm1.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_84 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_83, &detector_tmp_84)?
    })? {
        // comfy/model_detection.py:212
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:213
        cx.set(
            "in_w",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}img_in.proj.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:214
        cx.set(
            "out_w",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}final_layer.linear.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:215
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("hunyuan_video"),
        )?;
        // comfy/model_detection.py:216
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.index(&cx.attribute(&cx.get("in_w")?, "shape")?, &integer(1_i128)?)?,
        )?;
        // comfy/model_detection.py:217
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            cx.function(
                "list",
                vec![cx.slice(
                    &cx.attribute(&cx.get("in_w")?, "shape")?,
                    Some(integer(2_i128)?),
                    None,
                    None,
                )?],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:218
        cx.set_item(
            &cx.get("dit_config")?,
            &text("out_channels"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(&cx.get("out_w")?, "shape")?,
                    &integer(0_i128)?,
                )?,
                &cx.function(
                    "math.prod",
                    vec![cx.index(&cx.get("dit_config")?, &text("patch_size"))?],
                    vec![],
                )?,
            )?,
        )?;
        // comfy/model_detection.py:219
        if cx.truth(&cx.comprehension(
            cx.iter(&cx.get("state_dict_keys")?)?,
            "any",
            |cx, item| {
                cx.set("s", item);
                Ok(Some(cx.method(
                    &cx.get("s")?,
                    "startswith",
                    vec![cx.method(
                        &text("{}vector_in."),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?],
                    vec![],
                )?))
            },
        )?)? {
            // comfy/model_detection.py:220
            cx.set_item(
                &cx.get("dit_config")?,
                &text("vec_in_dim"),
                integer(768_i128)?,
            )?;
        } else {
            // comfy/model_detection.py:222
            cx.set_item(&cx.get("dit_config")?, &text("vec_in_dim"), none())?;
        }
        // comfy/model_detection.py:224
        if cx.truth(&{
            let detector_tmp_85 = cx.function(
                "len",
                vec![cx.index(&cx.get("dit_config")?, &text("patch_size"))?],
                vec![],
            )?;
            let detector_tmp_86 = integer(2_i128)?;
            cx.compare("Eq", &detector_tmp_85, &detector_tmp_86)?
        })? {
            // comfy/model_detection.py:225
            cx.set_item(
                &cx.get("dit_config")?,
                &text("axes_dim"),
                list(vec![integer(64_i128)?, integer(64_i128)?]),
            )?;
        } else {
            // comfy/model_detection.py:227
            cx.set_item(
                &cx.get("dit_config")?,
                &text("axes_dim"),
                list(vec![
                    integer(16_i128)?,
                    integer(56_i128)?,
                    integer(56_i128)?,
                ]),
            )?;
        }
        // comfy/model_detection.py:229
        if cx.truth(&cx.comprehension(
            cx.iter(&cx.get("state_dict_keys")?)?,
            "any",
            |cx, item| {
                cx.set("s", item);
                Ok(Some(cx.method(
                    &cx.get("s")?,
                    "startswith",
                    vec![cx.method(
                        &text("{}time_r_in."),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?],
                    vec![],
                )?))
            },
        )?)? {
            // comfy/model_detection.py:230
            cx.set_item(&cx.get("dit_config")?, &text("meanflow"), boolean(true))?;
        } else {
            // comfy/model_detection.py:232
            cx.set_item(&cx.get("dit_config")?, &text("meanflow"), boolean(false))?;
        }
        // comfy/model_detection.py:234
        cx.set_item(
            &cx.get("dit_config")?,
            &text("context_in_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}txt_in.input_embedder.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:235
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            cx.index(&cx.attribute(&cx.get("in_w")?, "shape")?, &integer(0_i128)?)?,
        )?;
        // comfy/model_detection.py:236
        cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio"), float(4.0_f64))?;
        // comfy/model_detection.py:237
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(&cx.attribute(&cx.get("in_w")?, "shape")?, &integer(0_i128)?)?,
                &integer(128_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:238
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}double_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:239
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth_single_blocks"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}single_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:240
        cx.set_item(&cx.get("dit_config")?, &text("theta"), integer(256_i128)?)?;
        // comfy/model_detection.py:241
        cx.set_item(&cx.get("dit_config")?, &text("qkv_bias"), boolean(true))?;
        // comfy/model_detection.py:242
        if cx.truth(&{
            let detector_tmp_87 = cx.method(
                &text("{}byt5_in.fc1.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_88 = cx.get("state_dict")?;
            cx.compare("In", &detector_tmp_87, &detector_tmp_88)?
        })? {
            // comfy/model_detection.py:243
            cx.set_item(&cx.get("dit_config")?, &text("byt5"), boolean(true))?;
        } else {
            // comfy/model_detection.py:245
            cx.set_item(&cx.get("dit_config")?, &text("byt5"), boolean(false))?;
        }
        // comfy/model_detection.py:247
        cx.set(
            "guidance_keys",
            cx.function(
                "list",
                vec![cx.comprehension(
                    cx.iter(&cx.get("state_dict_keys")?)?,
                    "list",
                    |cx, item| {
                        cx.set("a", item);
                        if !cx.truth(&cx.method(
                            &cx.get("a")?,
                            "startswith",
                            vec![cx.method(
                                &text("{}guidance_in."),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?],
                            vec![],
                        )?)? {
                            return Ok(None);
                        }
                        Ok(Some(cx.get("a")?))
                    },
                )?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:248
        cx.set_item(&cx.get("dit_config")?, &text("guidance_embed"), {
            let detector_tmp_89 = cx.function("len", vec![cx.get("guidance_keys")?], vec![])?;
            let detector_tmp_90 = integer(0_i128)?;
            cx.compare("Gt", &detector_tmp_89, &detector_tmp_90)?
        })?;
        // comfy/model_detection.py:251
        if cx.truth(&{
            let detector_tmp_91 = cx.method(
                &text("{}cond_type_embedding.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_92 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_91, &detector_tmp_92)?
        })? {
            // comfy/model_detection.py:252
            cx.set_item(
                &cx.get("dit_config")?,
                &text("use_cond_type_embedding"),
                boolean(true),
            )?;
        } else {
            // comfy/model_detection.py:254
            cx.set_item(
                &cx.get("dit_config")?,
                &text("use_cond_type_embedding"),
                boolean(false),
            )?;
        }
        // comfy/model_detection.py:255
        if cx.truth(&{
            let detector_tmp_93 = cx.method(
                &text("{}vision_in.proj.0.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_94 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_93, &detector_tmp_94)?
        })? {
            // comfy/model_detection.py:256
            cx.set_item(
                &cx.get("dit_config")?,
                &text("vision_in_dim"),
                cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}vision_in.proj.0.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
            )?;
            // comfy/model_detection.py:257
            cx.set_item(&cx.get("dit_config")?, &text("meanflow_sum"), boolean(true))?;
        } else {
            // comfy/model_detection.py:259
            cx.set_item(&cx.get("dit_config")?, &text("vision_in_dim"), none())?;
            // comfy/model_detection.py:260
            cx.set_item(
                &cx.get("dit_config")?,
                &text("meanflow_sum"),
                boolean(false),
            )?;
        }
        // comfy/model_detection.py:261
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:263
    if cx.truth(&{
        let detector_tmp_97 = {
            let detector_tmp_98 = cx.method(
                &text("{}content_mode_blocks.image.0.attn_out.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_99 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_98, &detector_tmp_99)?
        };
        if cx.truth(&detector_tmp_97)? {
            {
                let detector_tmp_95 = cx.method(
                    &text("{}single_stream_modulations.txt.lin.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_96 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_95, &detector_tmp_96)?
            }
        } else {
            detector_tmp_97
        }
    })? {
        // comfy/model_detection.py:264
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:265
        cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("flux3"))?;
        // comfy/model_detection.py:266
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}content_mode_blocks.image.0.attn_out.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:267
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(&cx.get("dit_config")?, &text("hidden_size"))?,
                &integer(128_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:268
        cx.set_item(&cx.get("dit_config")?, &text("fused_qkv"), {
            let detector_tmp_100 = cx.method(
                &text("{}single_blocks.0.qkv.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_101 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_100, &detector_tmp_101)?
        })?;
        // comfy/model_detection.py:269
        cx.set_item(&cx.get("dit_config")?, &text("flow_map"), {
            let detector_tmp_102 = cx.method(
                &text("{}ref_time_in.in_layer.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_103 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_102, &detector_tmp_103)?
        })?;
        // comfy/model_detection.py:270
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}txt_mode_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:271
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth_single_blocks"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}single_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:272
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:274
    if cx.truth(&{
        let detector_tmp_107 = cx.function(
            "any_suffix_in",
            vec![
                cx.get("state_dict_keys")?,
                cx.get("key_prefix")?,
                text("double_blocks.0.img_attn.norm.key_norm."),
                list(vec![text("weight"), text("scale")]),
            ],
            vec![],
        )?;
        if cx.truth(&detector_tmp_107)? {
            {
                let detector_tmp_104 = {
                    let detector_tmp_105 = cx.method(
                        &text("{}img_in.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_106 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_105, &detector_tmp_106)?
                };
                if cx.truth(&detector_tmp_104)? {
                    detector_tmp_104
                } else {
                    cx.function(
                        "any_suffix_in",
                        vec![
                            cx.get("state_dict_keys")?,
                            cx.get("key_prefix")?,
                            text("distilled_guidance_layer.norms.0."),
                            list(vec![text("weight"), text("scale")]),
                        ],
                        vec![],
                    )?
                }
            }
        } else {
            detector_tmp_107
        }
    })? {
        // comfy/model_detection.py:275
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:276
        if cx.truth(&{
            let detector_tmp_108 = cx.method(
                &text("{}double_stream_modulation_img.lin.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_109 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_108, &detector_tmp_109)?
        })? {
            // comfy/model_detection.py:277
            cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("flux2"))?;
            // comfy/model_detection.py:278
            cx.set_item(
                &cx.get("dit_config")?,
                &text("axes_dim"),
                list(vec![
                    integer(32_i128)?,
                    integer(32_i128)?,
                    integer(32_i128)?,
                    integer(32_i128)?,
                ]),
            )?;
            // comfy/model_detection.py:279
            cx.set_item(
                &cx.get("dit_config")?,
                &text("num_heads"),
                integer(48_i128)?,
            )?;
            // comfy/model_detection.py:280
            cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio"), float(3.0_f64))?;
            // comfy/model_detection.py:281
            cx.set_item(&cx.get("dit_config")?, &text("theta"), integer(2000_i128)?)?;
            // comfy/model_detection.py:282
            cx.set_item(
                &cx.get("dit_config")?,
                &text("out_channels"),
                integer(128_i128)?,
            )?;
            // comfy/model_detection.py:283
            cx.set_item(
                &cx.get("dit_config")?,
                &text("global_modulation"),
                boolean(true),
            )?;
            // comfy/model_detection.py:284
            cx.set_item(&cx.get("dit_config")?, &text("mlp_silu_act"), boolean(true))?;
            // comfy/model_detection.py:285
            cx.set_item(&cx.get("dit_config")?, &text("qkv_bias"), boolean(false))?;
            // comfy/model_detection.py:286
            cx.set_item(&cx.get("dit_config")?, &text("ops_bias"), boolean(false))?;
            // comfy/model_detection.py:287
            cx.set_item(
                &cx.get("dit_config")?,
                &text("default_ref_method"),
                text("index"),
            )?;
            // comfy/model_detection.py:288
            cx.set_item(
                &cx.get("dit_config")?,
                &text("ref_index_scale"),
                float(10.0_f64),
            )?;
            // comfy/model_detection.py:289
            cx.set_item(
                &cx.get("dit_config")?,
                &text("txt_ids_dims"),
                list(vec![integer(3_i128)?]),
            )?;
            // comfy/model_detection.py:290
            cx.set("patch_size", integer(1_i128)?);
        } else {
            // comfy/model_detection.py:292
            cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("flux"))?;
            // comfy/model_detection.py:293
            cx.set_item(
                &cx.get("dit_config")?,
                &text("axes_dim"),
                list(vec![
                    integer(16_i128)?,
                    integer(56_i128)?,
                    integer(56_i128)?,
                ]),
            )?;
            // comfy/model_detection.py:294
            cx.set_item(
                &cx.get("dit_config")?,
                &text("num_heads"),
                integer(24_i128)?,
            )?;
            // comfy/model_detection.py:295
            cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio"), float(4.0_f64))?;
            // comfy/model_detection.py:296
            cx.set_item(&cx.get("dit_config")?, &text("theta"), integer(10000_i128)?)?;
            // comfy/model_detection.py:297
            cx.set_item(
                &cx.get("dit_config")?,
                &text("out_channels"),
                integer(16_i128)?,
            )?;
            // comfy/model_detection.py:298
            cx.set_item(&cx.get("dit_config")?, &text("qkv_bias"), boolean(true))?;
            // comfy/model_detection.py:299
            cx.set_item(&cx.get("dit_config")?, &text("txt_ids_dims"), list(vec![]))?;
            // comfy/model_detection.py:300
            cx.set("patch_size", integer(2_i128)?);
        }
        // comfy/model_detection.py:302
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:303
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            integer(3072_i128)?,
        )?;
        // comfy/model_detection.py:304
        cx.set_item(
            &cx.get("dit_config")?,
            &text("context_in_dim"),
            integer(4096_i128)?,
        )?;
        // comfy/model_detection.py:306
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            cx.get("patch_size")?,
        )?;
        // comfy/model_detection.py:307
        cx.set(
            "in_key",
            cx.method(
                &text("{}img_in.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:308
        if cx.truth(&{
            let detector_tmp_110 = cx.get("in_key")?;
            let detector_tmp_111 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_110, &detector_tmp_111)?
        })? {
            // comfy/model_detection.py:309
            cx.set("w", cx.index(&cx.get("state_dict")?, &cx.get("in_key")?)?);
            // comfy/model_detection.py:310
            cx.set_item(
                &cx.get("dit_config")?,
                &text("in_channels"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(1_i128)?)?,
                    &cx.binary("Mult", &cx.get("patch_size")?, &cx.get("patch_size")?)?,
                )?,
            )?;
            // comfy/model_detection.py:311
            cx.set_item(
                &cx.get("dit_config")?,
                &text("hidden_size"),
                cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(0_i128)?)?,
            )?;
        }
        // comfy/model_detection.py:313
        cx.set(
            "txt_in_key",
            cx.method(
                &text("{}txt_in.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:314
        if cx.truth(&{
            let detector_tmp_112 = cx.get("txt_in_key")?;
            let detector_tmp_113 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_112, &detector_tmp_113)?
        })? {
            // comfy/model_detection.py:315
            cx.set(
                "w",
                cx.index(&cx.get("state_dict")?, &cx.get("txt_in_key")?)?,
            );
            // comfy/model_detection.py:316
            cx.set_item(
                &cx.get("dit_config")?,
                &text("context_in_dim"),
                cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(1_i128)?)?,
            )?;
            // comfy/model_detection.py:317
            cx.set_item(
                &cx.get("dit_config")?,
                &text("hidden_size"),
                cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(0_i128)?)?,
            )?;
        }
        // comfy/model_detection.py:319
        cx.set(
            "vec_in_key",
            cx.method(
                &text("{}vector_in.in_layer.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:320
        if cx.truth(&{
            let detector_tmp_114 = cx.get("vec_in_key")?;
            let detector_tmp_115 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_114, &detector_tmp_115)?
        })? {
            // comfy/model_detection.py:321
            cx.set_item(
                &cx.get("dit_config")?,
                &text("vec_in_dim"),
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("vec_in_key")?)?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            )?;
        } else {
            // comfy/model_detection.py:323
            cx.set_item(&cx.get("dit_config")?, &text("vec_in_dim"), none())?;
        }
        // comfy/model_detection.py:325
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(&cx.get("dit_config")?, &text("hidden_size"))?,
                &cx.function(
                    "sum",
                    vec![cx.index(&cx.get("dit_config")?, &text("axes_dim"))?],
                    vec![],
                )?,
            )?,
        )?;
        // comfy/model_detection.py:327
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}double_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:328
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth_single_blocks"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}single_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:330
        if cx.truth(&{
            let detector_tmp_116 = cx.function(
                "any_suffix_in",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.get("key_prefix")?,
                    text("distilled_guidance_layer.0.norms.0."),
                    list(vec![text("weight"), text("scale")]),
                ],
                vec![],
            )?;
            if cx.truth(&detector_tmp_116)? {
                detector_tmp_116
            } else {
                cx.function(
                    "any_suffix_in",
                    vec![
                        cx.get("state_dict_keys")?,
                        cx.get("key_prefix")?,
                        text("distilled_guidance_layer.norms.0."),
                        list(vec![text("weight"), text("scale")]),
                    ],
                    vec![],
                )?
            }
        })? {
            // comfy/model_detection.py:331
            cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("chroma"))?;
            // comfy/model_detection.py:332
            cx.set_item(
                &cx.get("dit_config")?,
                &text("in_channels"),
                integer(64_i128)?,
            )?;
            // comfy/model_detection.py:333
            cx.set_item(
                &cx.get("dit_config")?,
                &text("out_channels"),
                integer(64_i128)?,
            )?;
            // comfy/model_detection.py:334
            cx.set_item(&cx.get("dit_config")?, &text("in_dim"), integer(64_i128)?)?;
            // comfy/model_detection.py:335
            cx.set_item(
                &cx.get("dit_config")?,
                &text("out_dim"),
                integer(3072_i128)?,
            )?;
            // comfy/model_detection.py:336
            cx.set_item(
                &cx.get("dit_config")?,
                &text("hidden_dim"),
                integer(5120_i128)?,
            )?;
            // comfy/model_detection.py:337
            cx.set_item(&cx.get("dit_config")?, &text("n_layers"), integer(5_i128)?)?;
            // comfy/model_detection.py:339
            if cx.truth(&cx.function(
                "any_suffix_in",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.get("key_prefix")?,
                    text("nerf_blocks.0.norm."),
                    list(vec![text("weight"), text("scale")]),
                ],
                vec![],
            )?)? {
                // comfy/model_detection.py:340
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("image_model"),
                    text("chroma_radiance"),
                )?;
                // comfy/model_detection.py:341
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("in_channels"),
                    integer(3_i128)?,
                )?;
                // comfy/model_detection.py:342
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("out_channels"),
                    integer(3_i128)?,
                )?;
                // comfy/model_detection.py:343
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("patch_size"),
                    cx.method(
                        &cx.method(
                            &cx.get("state_dict")?,
                            "get",
                            vec![cx.method(
                                &text("{}img_in_patch.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?],
                            vec![],
                        )?,
                        "size",
                        vec![],
                        vec![("dim", cx.unary("USub", &integer(1_i128)?)?)],
                    )?,
                )?;
                // comfy/model_detection.py:344
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("nerf_hidden_size"),
                    integer(64_i128)?,
                )?;
                // comfy/model_detection.py:345
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("nerf_mlp_ratio"),
                    integer(4_i128)?,
                )?;
                // comfy/model_detection.py:346
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("nerf_depth"),
                    integer(4_i128)?,
                )?;
                // comfy/model_detection.py:347
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("nerf_max_freqs"),
                    integer(8_i128)?,
                )?;
                // comfy/model_detection.py:348
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("nerf_tile_size"),
                    integer(512_i128)?,
                )?;
                // comfy/model_detection.py:349
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("nerf_final_head_type"),
                    if cx.truth(&cx.function(
                        "any_suffix_in",
                        vec![
                            cx.get("state_dict_keys")?,
                            cx.get("key_prefix")?,
                            text("nerf_final_layer_conv.norm."),
                            list(vec![text("weight"), text("scale")]),
                        ],
                        vec![],
                    )?)? {
                        text("conv")
                    } else {
                        text("linear")
                    },
                )?;
                // comfy/model_detection.py:350
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("nerf_embedder_dtype"),
                    cx.attribute(&cx.get("torch")?, "float32")?,
                )?;
                // comfy/model_detection.py:351
                if cx.truth(&{
                    let detector_tmp_117 = cx.method(
                        &text("{}__x0__"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_118 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_117, &detector_tmp_118)?
                })? {
                    // comfy/model_detection.py:352
                    cx.set_item(&cx.get("dit_config")?, &text("use_x0"), boolean(true))?;
                } else {
                    // comfy/model_detection.py:354
                    cx.set_item(&cx.get("dit_config")?, &text("use_x0"), boolean(false))?;
                }
                // comfy/model_detection.py:355
                if cx.truth(&{
                    let detector_tmp_119 = cx.method(
                        &text("{}__sequential__"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_120 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_119, &detector_tmp_120)?
                })? {
                    // comfy/model_detection.py:356
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("use_sequential_txt_ids"),
                        boolean(true),
                    )?;
                } else {
                    // comfy/model_detection.py:358
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("use_sequential_txt_ids"),
                        boolean(false),
                    )?;
                }
            }
        } else {
            // comfy/model_detection.py:360
            cx.set_item(&cx.get("dit_config")?, &text("guidance_embed"), {
                let detector_tmp_121 = cx.method(
                    &text("{}guidance_in.in_layer.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_122 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_121, &detector_tmp_122)?
            })?;
            // comfy/model_detection.py:361
            cx.set_item(&cx.get("dit_config")?, &text("yak_mlp"), {
                let detector_tmp_123 = cx.method(
                    &text("{}double_blocks.0.img_mlp.gate_proj.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_124 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_123, &detector_tmp_124)?
            })?;
            // comfy/model_detection.py:362
            cx.set_item(
                &cx.get("dit_config")?,
                &text("txt_norm"),
                cx.function(
                    "any_suffix_in",
                    vec![
                        cx.get("state_dict_keys")?,
                        cx.get("key_prefix")?,
                        text("txt_norm."),
                        list(vec![text("weight"), text("scale")]),
                    ],
                    vec![],
                )?,
            )?;
            // comfy/model_detection.py:363
            if cx.truth(&{
                let detector_tmp_125 = cx.index(&cx.get("dit_config")?, &text("yak_mlp"))?;
                if cx.truth(&detector_tmp_125)? {
                    cx.index(&cx.get("dit_config")?, &text("txt_norm"))?
                } else {
                    detector_tmp_125
                }
            })? {
                // comfy/model_detection.py:364
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("txt_ids_dims"),
                    list(vec![integer(1_i128)?, integer(2_i128)?]),
                )?;
            }
            // comfy/model_detection.py:365
            if cx.truth(&{
                let detector_tmp_128 = {
                    let detector_tmp_129 = cx.method(
                        &cx.get("dit_config")?,
                        "get",
                        vec![text("context_in_dim")],
                        vec![],
                    )?;
                    let detector_tmp_130 = integer(3584_i128)?;
                    cx.compare("Eq", &detector_tmp_129, &detector_tmp_130)?
                };
                if cx.truth(&detector_tmp_128)? {
                    {
                        let detector_tmp_126 =
                            cx.index(&cx.get("dit_config")?, &text("vec_in_dim"))?;
                        let detector_tmp_127 = none();
                        cx.compare("Is", &detector_tmp_126, &detector_tmp_127)?
                    }
                } else {
                    detector_tmp_128
                }
            })? {
                // comfy/model_detection.py:366
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("txt_ids_dims"),
                    list(vec![integer(1_i128)?, integer(2_i128)?]),
                )?;
            }
        }
        // comfy/model_detection.py:368
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:370
    if cx.truth(&{
        let detector_tmp_131 = cx.method(
            &text("{}t5_yproj.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_132 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_131, &detector_tmp_132)?
    })? {
        // comfy/model_detection.py:371
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:372
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("mochi_preview"),
        )?;
        // comfy/model_detection.py:373
        cx.set_item(&cx.get("dit_config")?, &text("depth"), integer(48_i128)?)?;
        // comfy/model_detection.py:374
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:375
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            integer(24_i128)?,
        )?;
        // comfy/model_detection.py:376
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size_x"),
            integer(3072_i128)?,
        )?;
        // comfy/model_detection.py:377
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size_y"),
            integer(1536_i128)?,
        )?;
        // comfy/model_detection.py:378
        cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio_x"), float(4.0_f64))?;
        // comfy/model_detection.py:379
        cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio_y"), float(4.0_f64))?;
        // comfy/model_detection.py:380
        cx.set_item(&cx.get("dit_config")?, &text("learn_sigma"), boolean(false))?;
        // comfy/model_detection.py:381
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(12_i128)?,
        )?;
        // comfy/model_detection.py:382
        cx.set_item(&cx.get("dit_config")?, &text("qk_norm"), boolean(true))?;
        // comfy/model_detection.py:383
        cx.set_item(&cx.get("dit_config")?, &text("qkv_bias"), boolean(false))?;
        // comfy/model_detection.py:384
        cx.set_item(&cx.get("dit_config")?, &text("out_bias"), boolean(true))?;
        // comfy/model_detection.py:385
        cx.set_item(&cx.get("dit_config")?, &text("attn_drop"), float(0.0_f64))?;
        // comfy/model_detection.py:386
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_embed_bias"),
            boolean(true),
        )?;
        // comfy/model_detection.py:387
        cx.set_item(
            &cx.get("dit_config")?,
            &text("posenc_preserve_area"),
            boolean(true),
        )?;
        // comfy/model_detection.py:388
        cx.set_item(
            &cx.get("dit_config")?,
            &text("timestep_mlp_bias"),
            boolean(true),
        )?;
        // comfy/model_detection.py:389
        cx.set_item(
            &cx.get("dit_config")?,
            &text("attend_to_padding"),
            boolean(false),
        )?;
        // comfy/model_detection.py:390
        cx.set_item(
            &cx.get("dit_config")?,
            &text("timestep_scale"),
            float(1000.0_f64),
        )?;
        // comfy/model_detection.py:391
        cx.set_item(&cx.get("dit_config")?, &text("use_t5"), boolean(true))?;
        // comfy/model_detection.py:392
        cx.set_item(
            &cx.get("dit_config")?,
            &text("t5_feat_dim"),
            integer(4096_i128)?,
        )?;
        // comfy/model_detection.py:393
        cx.set_item(
            &cx.get("dit_config")?,
            &text("t5_token_length"),
            integer(256_i128)?,
        )?;
        // comfy/model_detection.py:394
        cx.set_item(
            &cx.get("dit_config")?,
            &text("rope_theta"),
            float(10000.0_f64),
        )?;
        // comfy/model_detection.py:395
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:397
    if cx.truth(&{
        let detector_tmp_135 = {
            let detector_tmp_136 = cx.method(
                &text("{}adaln_single.emb.timestep_embedder.linear_1.bias"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_137 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_136, &detector_tmp_137)?
        };
        if cx.truth(&detector_tmp_135)? {
            {
                let detector_tmp_133 = cx.method(
                    &text("{}pos_embed.proj.bias"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_134 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_133, &detector_tmp_134)?
            }
        } else {
            detector_tmp_135
        }
    })? {
        // comfy/model_detection.py:399
        return Ok(none());
    }
    // comfy/model_detection.py:401
    if cx.truth(&{
        let detector_tmp_140 = {
            let detector_tmp_141 = cx.method(
                &text("{}video_patch_proj.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_142 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_141, &detector_tmp_142)?
        };
        if cx.truth(&detector_tmp_140)? {
            {
                let detector_tmp_138 = cx.method(
                    &text("{}audio_patch_proj.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_139 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_138, &detector_tmp_139)?
            }
        } else {
            detector_tmp_140
        }
    })? {
        // comfy/model_detection.py:402
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:403
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("minimax_h3"),
        )?;
        // comfy/model_detection.py:404
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:405
        cx.set_item(
            &cx.get("dit_config")?,
            &text("token_refiner_num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}token_refiner.blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:406
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}video_patch_proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:408
        cx.set("pdd_steps", integer(1_i128)?);
        // comfy/model_detection.py:409
        if cx.truth(&{
            let detector_tmp_145 = {
                let detector_tmp_146 = cx.get("metadata")?;
                let detector_tmp_147 = none();
                cx.compare("IsNot", &detector_tmp_146, &detector_tmp_147)?
            };
            if cx.truth(&detector_tmp_145)? {
                {
                    let detector_tmp_143 = text("config");
                    let detector_tmp_144 = cx.get("metadata")?;
                    cx.compare("In", &detector_tmp_143, &detector_tmp_144)?
                }
            } else {
                detector_tmp_145
            }
        })? {
            // comfy/model_detection.py:410
            cx.set(
                "pdd_steps",
                cx.method(
                    &cx.method(
                        &cx.function(
                            "json.loads",
                            vec![cx.index(&cx.get("metadata")?, &text("config"))?],
                            vec![],
                        )?,
                        "get",
                        vec![text("transformer"), cx.mapping(vec![])?],
                        vec![],
                    )?,
                    "get",
                    vec![text("pdd_steps"), integer(1_i128)?],
                    vec![],
                )?,
            );
        }
        // comfy/model_detection.py:411
        cx.set_item(
            &cx.get("dit_config")?,
            &text("latents_dim"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}final_layer.video_out.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &cx.binary("Mult", &integer(4_i128)?, &cx.get("pdd_steps")?)?,
            )?,
        )?;
        // comfy/model_detection.py:412
        cx.set_item(
            &cx.get("dit_config")?,
            &text("audio_latents_dim"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}final_layer.audio_out.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &cx.get("pdd_steps")?,
            )?,
        )?;
        // comfy/model_detection.py:413
        cx.set_item(
            &cx.get("dit_config")?,
            &text("attention_head_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}blocks.0.attn.q_norm.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:414
        cx.set(
            "qkv",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}blocks.0.attn.qkv_proj.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:415
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_attention_heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(&cx.attribute(&cx.get("qkv")?, "shape")?, &integer(0_i128)?)?,
                &cx.binary(
                    "Mult",
                    &integer(3_i128)?,
                    &cx.index(&cx.get("dit_config")?, &text("attention_head_dim"))?,
                )?,
            )?,
        )?;
        // comfy/model_detection.py:416
        cx.set_item(
            &cx.get("dit_config")?,
            &text("ffn_hidden_size"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}blocks.0.mlp.fc1.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &integer(2_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:417
        cx.set_item(
            &cx.get("dit_config")?,
            &text("text_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}condition_proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:418
        cx.set(
            "table_key",
            cx.method(
                &text("{}adaln_t_table"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:419
        if cx.truth(&{
            let detector_tmp_148 = cx.get("table_key")?;
            let detector_tmp_149 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_148, &detector_tmp_149)?
        })? {
            // comfy/model_detection.py:421
            cx.set(
                "table",
                cx.attribute(
                    &cx.index(&cx.get("state_dict")?, &cx.get("table_key")?)?,
                    "shape",
                )?,
            );
            // comfy/model_detection.py:422
            cx.set_item(
                &cx.get("dit_config")?,
                &text("adaln_curve_grid"),
                cx.index(&cx.get("table")?, &integer(0_i128)?)?,
            )?;
            // comfy/model_detection.py:423
            cx.set_item(
                &cx.get("dit_config")?,
                &text("time_embed_dim"),
                cx.index(&cx.get("table")?, &integer(1_i128)?)?,
            )?;
        } else {
            // comfy/model_detection.py:425
            cx.set(
                "te",
                cx.index(
                    &cx.get("state_dict")?,
                    &cx.method(
                        &text("{}time_embedder.proj_in.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                )?,
            );
            // comfy/model_detection.py:426
            cx.set_item(
                &cx.get("dit_config")?,
                &text("timestep_input_dim"),
                cx.index(&cx.attribute(&cx.get("te")?, "shape")?, &integer(1_i128)?)?,
            )?;
            // comfy/model_detection.py:427
            cx.set_item(
                &cx.get("dit_config")?,
                &text("time_embed_hidden_size"),
                cx.index(&cx.attribute(&cx.get("te")?, "shape")?, &integer(0_i128)?)?,
            )?;
            // comfy/model_detection.py:428
            cx.set_item(
                &cx.get("dit_config")?,
                &text("time_embed_dim"),
                cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}time_embedder.proj_out.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:429
        cx.set_item(
            &cx.get("dit_config")?,
            &text("rope_inv_freq_len"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}rope.inv_freq"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:430
        cx.set_item(&cx.get("dit_config")?, &text("gate_compress"), {
            let detector_tmp_150 = cx.method(
                &text("{}blocks.0.attn.to_gate_compress.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_151 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_150, &detector_tmp_151)?
        })?;
        // comfy/model_detection.py:431
        if cx.truth(&{
            let detector_tmp_154 = {
                let detector_tmp_155 = cx.get("metadata")?;
                let detector_tmp_156 = none();
                cx.compare("IsNot", &detector_tmp_155, &detector_tmp_156)?
            };
            if cx.truth(&detector_tmp_154)? {
                {
                    let detector_tmp_152 = text("config");
                    let detector_tmp_153 = cx.get("metadata")?;
                    cx.compare("In", &detector_tmp_152, &detector_tmp_153)?
                }
            } else {
                detector_tmp_154
            }
        })? {
            // comfy/model_detection.py:432
            cx.method(
                &cx.get("dit_config")?,
                "update",
                vec![cx.method(
                    &cx.function(
                        "json.loads",
                        vec![cx.index(&cx.get("metadata")?, &text("config"))?],
                        vec![],
                    )?,
                    "get",
                    vec![text("transformer"), cx.mapping(vec![])?],
                    vec![],
                )?],
                vec![],
            )?;
        }
        // comfy/model_detection.py:433
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:435
    if cx.truth(&{
        let detector_tmp_157 = cx.method(
            &text("{}adaln_single.emb.timestep_embedder.linear_1.bias"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_158 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_157, &detector_tmp_158)?
    })? {
        // comfy/model_detection.py:436
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:437
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            if cx.truth(&{
                let detector_tmp_159 = cx.concatenate_text(&[
                    cx.string(&cx.get("key_prefix")?)?,
                    text("audio_adaln_single.linear.weight"),
                ])?;
                let detector_tmp_160 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_159, &detector_tmp_160)?
            })? {
                text("ltxav")
            } else {
                text("ltxv")
            },
        )?;
        // comfy/model_detection.py:438
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}transformer_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:439
        cx.set(
            "shape",
            cx.attribute(
                &cx.index(
                    &cx.get("state_dict")?,
                    &cx.method(
                        &text("{}transformer_blocks.0.attn2.to_k.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                )?,
                "shape",
            )?,
        );
        // comfy/model_detection.py:440
        cx.set_item(
            &cx.get("dit_config")?,
            &text("attention_head_dim"),
            cx.binary(
                "FloorDiv",
                &cx.index(&cx.get("shape")?, &integer(0_i128)?)?,
                &integer(32_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:441
        cx.set_item(
            &cx.get("dit_config")?,
            &text("cross_attention_dim"),
            cx.index(&cx.get("shape")?, &integer(1_i128)?)?,
        )?;
        // comfy/model_detection.py:442
        if cx.truth(&{
            let detector_tmp_163 = {
                let detector_tmp_164 = cx.get("metadata")?;
                let detector_tmp_165 = none();
                cx.compare("IsNot", &detector_tmp_164, &detector_tmp_165)?
            };
            if cx.truth(&detector_tmp_163)? {
                {
                    let detector_tmp_161 = text("config");
                    let detector_tmp_162 = cx.get("metadata")?;
                    cx.compare("In", &detector_tmp_161, &detector_tmp_162)?
                }
            } else {
                detector_tmp_163
            }
        })? {
            // comfy/model_detection.py:443
            cx.method(
                &cx.get("dit_config")?,
                "update",
                vec![cx.method(
                    &cx.function(
                        "json.loads",
                        vec![cx.index(&cx.get("metadata")?, &text("config"))?],
                        vec![],
                    )?,
                    "get",
                    vec![text("transformer"), cx.mapping(vec![])?],
                    vec![],
                )?],
                vec![],
            )?;
        }
        // comfy/model_detection.py:444
        cx.set_item(
            &cx.get("dit_config")?,
            &text("use_keyframes_abs_pos_embedding"),
            {
                let detector_tmp_166 = cx.method(
                    &text("{}keyframes_abs_pos_embedding"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_167 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_166, &detector_tmp_167)?
            },
        )?;
        // comfy/model_detection.py:445
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:447
    if cx.truth(&{
        let detector_tmp_168 = cx.method(
            &text("{}genre_embedder.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_169 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_168, &detector_tmp_169)?
    })? {
        // comfy/model_detection.py:448
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:449
        cx.set_item(&cx.get("dit_config")?, &text("audio_model"), text("ace"))?;
        // comfy/model_detection.py:450
        cx.set_item(
            &cx.get("dit_config")?,
            &text("attention_head_dim"),
            integer(128_i128)?,
        )?;
        // comfy/model_detection.py:451
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(8_i128)?,
        )?;
        // comfy/model_detection.py:452
        cx.set_item(
            &cx.get("dit_config")?,
            &text("inner_dim"),
            integer(2560_i128)?,
        )?;
        // comfy/model_detection.py:453
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_height"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:454
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_position"),
            integer(32768_i128)?,
        )?;
        // comfy/model_detection.py:455
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_width"),
            integer(32768_i128)?,
        )?;
        // comfy/model_detection.py:456
        cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio"), float(2.5_f64))?;
        // comfy/model_detection.py:457
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_attention_heads"),
            integer(20_i128)?,
        )?;
        // comfy/model_detection.py:458
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            integer(24_i128)?,
        )?;
        // comfy/model_detection.py:459
        cx.set_item(
            &cx.get("dit_config")?,
            &text("out_channels"),
            integer(8_i128)?,
        )?;
        // comfy/model_detection.py:460
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            list(vec![integer(16_i128)?, integer(1_i128)?]),
        )?;
        // comfy/model_detection.py:461
        cx.set_item(
            &cx.get("dit_config")?,
            &text("rope_theta"),
            float(1000000.0_f64),
        )?;
        // comfy/model_detection.py:462
        cx.set_item(
            &cx.get("dit_config")?,
            &text("speaker_embedding_dim"),
            integer(512_i128)?,
        )?;
        // comfy/model_detection.py:463
        cx.set_item(
            &cx.get("dit_config")?,
            &text("text_embedding_dim"),
            integer(768_i128)?,
        )?;
        // comfy/model_detection.py:465
        cx.set_item(
            &cx.get("dit_config")?,
            &text("ssl_encoder_depths"),
            list(vec![integer(8_i128)?, integer(8_i128)?]),
        )?;
        // comfy/model_detection.py:466
        cx.set_item(
            &cx.get("dit_config")?,
            &text("ssl_latent_dims"),
            list(vec![integer(1024_i128)?, integer(768_i128)?]),
        )?;
        // comfy/model_detection.py:467
        cx.set_item(
            &cx.get("dit_config")?,
            &text("ssl_names"),
            list(vec![text("mert"), text("m-hubert")]),
        )?;
        // comfy/model_detection.py:468
        cx.set_item(
            &cx.get("dit_config")?,
            &text("lyric_encoder_vocab_size"),
            integer(6693_i128)?,
        )?;
        // comfy/model_detection.py:469
        cx.set_item(
            &cx.get("dit_config")?,
            &text("lyric_hidden_size"),
            integer(1024_i128)?,
        )?;
        // comfy/model_detection.py:470
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:472
    if cx.truth(&{
        let detector_tmp_170 = cx.method(
            &text("{}t_block.1.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_171 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_170, &detector_tmp_171)?
    })? {
        // comfy/model_detection.py:473
        cx.set("patch_size", integer(2_i128)?);
        // comfy/model_detection.py:474
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:475
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:476
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            cx.get("patch_size")?,
        )?;
        // comfy/model_detection.py:477
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            integer(1152_i128)?,
        )?;
        // comfy/model_detection.py:478
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(4_i128)?,
        )?;
        // comfy/model_detection.py:479
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:481
        cx.set(
            "y_key",
            cx.method(
                &text("{}y_embedder.y_embedding"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:482
        if cx.truth(&{
            let detector_tmp_172 = cx.get("y_key")?;
            let detector_tmp_173 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_172, &detector_tmp_173)?
        })? {
            // comfy/model_detection.py:483
            cx.set_item(
                &cx.get("dit_config")?,
                &text("model_max_length"),
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("y_key")?)?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:485
        cx.set(
            "pe_key",
            cx.method(
                &text("{}pos_embed"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:486
        if cx.truth(&{
            let detector_tmp_174 = cx.get("pe_key")?;
            let detector_tmp_175 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_174, &detector_tmp_175)?
        })? {
            // comfy/model_detection.py:487
            cx.set_item(
                &cx.get("dit_config")?,
                &text("input_size"),
                cx.binary(
                    "Mult",
                    &cx.function(
                        "int",
                        vec![cx.function(
                            "math.sqrt",
                            vec![cx.index(
                                &cx.attribute(
                                    &cx.index(&cx.get("state_dict")?, &cx.get("pe_key")?)?,
                                    "shape",
                                )?,
                                &integer(1_i128)?,
                            )?],
                            vec![],
                        )?],
                        vec![],
                    )?,
                    &cx.get("patch_size")?,
                )?,
            )?;
            // comfy/model_detection.py:488
            cx.set_item(
                &cx.get("dit_config")?,
                &text("pe_interpolation"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(&cx.get("dit_config")?, &text("input_size"))?,
                    &cx.binary("FloorDiv", &integer(512_i128)?, &integer(8_i128)?)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:490
        cx.set(
            "ar_key",
            cx.method(
                &text("{}ar_embedder.mlp.0.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:491
        if cx.truth(&{
            let detector_tmp_176 = cx.get("ar_key")?;
            let detector_tmp_177 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_176, &detector_tmp_177)?
        })? {
            // comfy/model_detection.py:492
            cx.set_item(
                &cx.get("dit_config")?,
                &text("image_model"),
                text("pixart_alpha"),
            )?;
            // comfy/model_detection.py:493
            cx.set_item(
                &cx.get("dit_config")?,
                &text("micro_condition"),
                boolean(true),
            )?;
        } else {
            // comfy/model_detection.py:495
            cx.set_item(
                &cx.get("dit_config")?,
                &text("image_model"),
                text("pixart_sigma"),
            )?;
            // comfy/model_detection.py:496
            cx.set_item(
                &cx.get("dit_config")?,
                &text("micro_condition"),
                boolean(false),
            )?;
        }
        // comfy/model_detection.py:497
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:499
    if cx.truth(&{
        let detector_tmp_178 = cx.method(
            &text("{}blocks.block0.blocks.0.block.attn.to_q.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_179 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_178, &detector_tmp_179)?
    })? {
        // comfy/model_detection.py:500
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:501
        cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("cosmos"))?;
        // comfy/model_detection.py:502
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_img_h"),
            integer(240_i128)?,
        )?;
        // comfy/model_detection.py:503
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_img_w"),
            integer(240_i128)?,
        )?;
        // comfy/model_detection.py:504
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_frames"),
            integer(128_i128)?,
        )?;
        // comfy/model_detection.py:505
        cx.set("concat_padding_mask", boolean(true));
        // comfy/model_detection.py:506
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.binary(
                "Sub",
                &cx.binary(
                    "FloorDiv",
                    &cx.index(
                        &cx.attribute(
                            &cx.index(
                                &cx.get("state_dict")?,
                                &cx.method(
                                    &text("{}x_embedder.proj.1.weight"),
                                    "format",
                                    vec![cx.get("key_prefix")?],
                                    vec![],
                                )?,
                            )?,
                            "shape",
                        )?,
                        &integer(1_i128)?,
                    )?,
                    &integer(4_i128)?,
                )?,
                &cx.function("int", vec![cx.get("concat_padding_mask")?], vec![])?,
            )?,
        )?;
        // comfy/model_detection.py:507
        cx.set_item(
            &cx.get("dit_config")?,
            &text("out_channels"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:508
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_spatial"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:509
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_temporal"),
            integer(1_i128)?,
        )?;
        // comfy/model_detection.py:510
        cx.set_item(
            &cx.get("dit_config")?,
            &text("model_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}blocks.block0.blocks.0.block.attn.to_q.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:511
        cx.set_item(
            &cx.get("dit_config")?,
            &text("block_config"),
            text("FA-CA-MLP"),
        )?;
        // comfy/model_detection.py:512
        cx.set_item(
            &cx.get("dit_config")?,
            &text("concat_padding_mask"),
            cx.get("concat_padding_mask")?,
        )?;
        // comfy/model_detection.py:513
        cx.set_item(&cx.get("dit_config")?, &text("pos_emb_cls"), text("rope3d"))?;
        // comfy/model_detection.py:514
        cx.set_item(
            &cx.get("dit_config")?,
            &text("pos_emb_learnable"),
            boolean(false),
        )?;
        // comfy/model_detection.py:515
        cx.set_item(
            &cx.get("dit_config")?,
            &text("pos_emb_interpolation"),
            text("crop"),
        )?;
        // comfy/model_detection.py:516
        cx.set_item(
            &cx.get("dit_config")?,
            &text("block_x_format"),
            text("THWBD"),
        )?;
        // comfy/model_detection.py:517
        cx.set_item(
            &cx.get("dit_config")?,
            &text("affline_emb_norm"),
            boolean(true),
        )?;
        // comfy/model_detection.py:518
        cx.set_item(
            &cx.get("dit_config")?,
            &text("use_adaln_lora"),
            boolean(true),
        )?;
        // comfy/model_detection.py:519
        cx.set_item(
            &cx.get("dit_config")?,
            &text("adaln_lora_dim"),
            integer(256_i128)?,
        )?;
        // comfy/model_detection.py:521
        if cx.truth(&{
            let detector_tmp_180 = cx.index(&cx.get("dit_config")?, &text("model_channels"))?;
            let detector_tmp_181 = integer(4096_i128)?;
            cx.compare("Eq", &detector_tmp_180, &detector_tmp_181)?
        })? {
            // comfy/model_detection.py:523
            cx.set_item(
                &cx.get("dit_config")?,
                &text("num_blocks"),
                integer(28_i128)?,
            )?;
            // comfy/model_detection.py:524
            cx.set_item(
                &cx.get("dit_config")?,
                &text("num_heads"),
                integer(32_i128)?,
            )?;
            // comfy/model_detection.py:525
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_per_block_abs_pos_emb"),
                boolean(true),
            )?;
            // comfy/model_detection.py:526
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_h_extrapolation_ratio"),
                float(1.0_f64),
            )?;
            // comfy/model_detection.py:527
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_w_extrapolation_ratio"),
                float(1.0_f64),
            )?;
            // comfy/model_detection.py:528
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_t_extrapolation_ratio"),
                float(2.0_f64),
            )?;
            // comfy/model_detection.py:529
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_per_block_abs_pos_emb_type"),
                text("learnable"),
            )?;
        } else {
            // comfy/model_detection.py:532
            cx.set_item(
                &cx.get("dit_config")?,
                &text("num_blocks"),
                integer(36_i128)?,
            )?;
            // comfy/model_detection.py:533
            cx.set_item(
                &cx.get("dit_config")?,
                &text("num_heads"),
                integer(40_i128)?,
            )?;
            // comfy/model_detection.py:534
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_per_block_abs_pos_emb"),
                boolean(true),
            )?;
            // comfy/model_detection.py:535
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_h_extrapolation_ratio"),
                float(2.0_f64),
            )?;
            // comfy/model_detection.py:536
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_w_extrapolation_ratio"),
                float(2.0_f64),
            )?;
            // comfy/model_detection.py:537
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_t_extrapolation_ratio"),
                float(2.0_f64),
            )?;
            // comfy/model_detection.py:538
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_h_extrapolation_ratio"),
                float(2.0_f64),
            )?;
            // comfy/model_detection.py:539
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_w_extrapolation_ratio"),
                float(2.0_f64),
            )?;
            // comfy/model_detection.py:540
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_t_extrapolation_ratio"),
                float(2.0_f64),
            )?;
            // comfy/model_detection.py:541
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_per_block_abs_pos_emb_type"),
                text("learnable"),
            )?;
        }
        // comfy/model_detection.py:542
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:545
    cx.set(
        "_lq_w_key",
        cx.method(
            &text("{}lq_proj.latent_proj.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:546
    if cx.truth(&{
        let detector_tmp_182 = cx.get("_lq_w_key")?;
        let detector_tmp_183 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_182, &detector_tmp_183)?
    })? {
        // comfy/model_detection.py:547
        cx.set(
            "latent_proj_in_channels",
            cx.function(
                "int",
                vec![cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("_lq_w_key")?)?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:548
        cx.set(
            "hidden_dim",
            cx.function(
                "int",
                vec![cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("_lq_w_key")?)?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:549
        cx.set(
            "_gate_prefix",
            cx.method(
                &text("{}lq_proj.gate_modules."),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:550
        cx.set(
            "num_gates",
            cx.comprehension(
                cx.iter(&cx.get("state_dict_keys")?)?,
                "unique_len",
                |cx, item| {
                    cx.set("k", item);
                    if !cx.truth(&cx.method(
                        &cx.get("k")?,
                        "startswith",
                        vec![cx.get("_gate_prefix")?],
                        vec![],
                    )?)? {
                        return Ok(None);
                    }
                    Ok(Some(cx.index(
                        &cx.method(
                            &cx.slice(
                                &cx.get("k")?,
                                Some(cx.function("len", vec![cx.get("_gate_prefix")?], vec![])?),
                                None,
                                None,
                            )?,
                            "split",
                            vec![text(".")],
                            vec![],
                        )?,
                        &integer(0_i128)?,
                    )?))
                },
            )?,
        );
        // comfy/model_detection.py:552
        cx.set("pid_v1_5", {
            let detector_tmp_184 = cx.method(
                &text("{}lq_proj.pit_head.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_185 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_184, &detector_tmp_185)?
        });
        // comfy/model_detection.py:553
        cx.set(
            "dit_config",
            cx.mapping(vec![
                (text("image_model"), text("pid")),
                (text("lq_hidden_dim"), cx.get("hidden_dim")?),
            ])?,
        );
        // comfy/model_detection.py:555
        if cx.truth(&{
            let detector_tmp_186 = cx.get("num_gates")?;
            let detector_tmp_187 = integer(0_i128)?;
            cx.compare("Gt", &detector_tmp_186, &detector_tmp_187)?
        })? {
            // comfy/model_detection.py:556
            cx.set_item(
                &cx.get("dit_config")?,
                &text("lq_interval"),
                cx.binary(
                    "FloorDiv",
                    &cx.binary(
                        "Sub",
                        &cx.binary("Add", &integer(14_i128)?, &cx.get("num_gates")?)?,
                        &integer(1_i128)?,
                    )?,
                    &cx.get("num_gates")?,
                )?,
            )?;
        }
        // comfy/model_detection.py:557
        if cx.truth(&cx.get("pid_v1_5")?)? {
            // comfy/model_detection.py:558
            cx.set(
                "pid_v1_5_variants",
                cx.mapping(vec![
                    (
                        integer(16_i128)?,
                        cx.mapping(vec![
                            (text("lq_latent_channels"), integer(16_i128)?),
                            (text("latent_spatial_down_factor"), integer(8_i128)?),
                            (text("lq_latent_unpatchify_factor"), integer(1_i128)?),
                        ])?,
                    ),
                    (
                        integer(32_i128)?,
                        cx.mapping(vec![
                            (text("lq_latent_channels"), integer(128_i128)?),
                            (text("latent_spatial_down_factor"), integer(16_i128)?),
                            (text("lq_latent_unpatchify_factor"), integer(2_i128)?),
                        ])?,
                    ),
                ])?,
            );
            // comfy/model_detection.py:570
            cx.set(
                "variant",
                cx.method(
                    &cx.get("pid_v1_5_variants")?,
                    "get",
                    vec![cx.get("latent_proj_in_channels")?],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:571
            if cx.truth(&{
                let detector_tmp_188 = cx.get("variant")?;
                let detector_tmp_189 = none();
                cx.compare("Is", &detector_tmp_188, &detector_tmp_189)?
            })? {
                // comfy/model_detection.py:572
                return Err(cx.exception(
                    "ValueError",
                    &cx.concatenate_text(&[
                        text("Unsupported PiD v1.5 latent projection with "),
                        cx.string(&cx.get("latent_proj_in_channels")?)?,
                        text(" input channels"),
                    ])?,
                )?);
            }
            // comfy/model_detection.py:573
            cx.set(
                "gate_weight",
                cx.index(
                    &cx.get("state_dict")?,
                    &cx.method(
                        &text("{}lq_proj.gate_modules.0.content_proj.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                )?,
            );
            // comfy/model_detection.py:574
            cx.method(
                &cx.get("dit_config")?,
                "update",
                vec![cx.get("variant")?],
                vec![],
            )?;
            // comfy/model_detection.py:575
            cx.method(
                &cx.get("dit_config")?,
                "update",
                vec![cx.mapping(vec![
                    (text("lq_conv_padding_mode"), text("replicate")),
                    (text("lq_gate_per_token"), {
                        let detector_tmp_190 = cx.index(
                            &cx.attribute(&cx.get("gate_weight")?, "shape")?,
                            &integer(0_i128)?,
                        )?;
                        let detector_tmp_191 = integer(1_i128)?;
                        cx.compare("Eq", &detector_tmp_190, &detector_tmp_191)?
                    }),
                    (text("pit_lq_inject"), boolean(true)),
                    (text("rope_ref_h"), integer(2048_i128)?),
                    (text("rope_ref_w"), integer(2048_i128)?),
                ])?],
                vec![],
            )?;
        } else {
            // comfy/model_detection.py:583
            cx.method(
                &cx.get("dit_config")?,
                "update",
                vec![cx.mapping(vec![
                    (
                        text("lq_latent_channels"),
                        cx.get("latent_proj_in_channels")?,
                    ),
                    (
                        text("latent_spatial_down_factor"),
                        if cx.truth(&{
                            let detector_tmp_192 = cx.get("latent_proj_in_channels")?;
                            let detector_tmp_193 = integer(64_i128)?;
                            cx.compare("GtE", &detector_tmp_192, &detector_tmp_193)?
                        })? {
                            integer(16_i128)?
                        } else {
                            integer(8_i128)?
                        },
                    ),
                ])?],
                vec![],
            )?;
        }
        // comfy/model_detection.py:587
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:589
    if cx.truth(&{
        let detector_tmp_194 = cx.method(
            &text("{}core.pixel_embedder.proj.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_195 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_194, &detector_tmp_195)?
    })? {
        // comfy/model_detection.py:590
        return cx.mapping(vec![(text("image_model"), text("pixeldit_t2i"))]);
    }
    // comfy/model_detection.py:592
    if cx.truth(&{
        let detector_tmp_198 = {
            let detector_tmp_199 = cx.method(
                &text("{}cap_embedder.1.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_200 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_199, &detector_tmp_200)?
        };
        if cx.truth(&detector_tmp_198)? {
            {
                let detector_tmp_196 = cx.method(
                    &text("{}noise_refiner.0.attention.k_norm.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_197 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_196, &detector_tmp_197)?
            }
        } else {
            detector_tmp_198
        }
    })? {
        // comfy/model_detection.py:593
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:594
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("lumina2"),
        )?;
        // comfy/model_detection.py:595
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:596
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:597
        cx.set(
            "w",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}cap_embedder.1.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:598
        cx.set_item(
            &cx.get("dit_config")?,
            &text("dim"),
            cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(0_i128)?)?,
        )?;
        // comfy/model_detection.py:599
        cx.set_item(
            &cx.get("dit_config")?,
            &text("cap_feat_dim"),
            cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(1_i128)?)?,
        )?;
        // comfy/model_detection.py:600
        cx.set_item(
            &cx.get("dit_config")?,
            &text("n_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:601
        cx.set_item(&cx.get("dit_config")?, &text("qk_norm"), boolean(true))?;
        // comfy/model_detection.py:603
        if cx.truth(&{
            let detector_tmp_201 = cx.index(&cx.get("dit_config")?, &text("dim"))?;
            let detector_tmp_202 = integer(2304_i128)?;
            cx.compare("Eq", &detector_tmp_201, &detector_tmp_202)?
        })? {
            // comfy/model_detection.py:604
            cx.set_item(&cx.get("dit_config")?, &text("n_heads"), integer(24_i128)?)?;
            // comfy/model_detection.py:605
            cx.set_item(
                &cx.get("dit_config")?,
                &text("n_kv_heads"),
                integer(8_i128)?,
            )?;
            // comfy/model_detection.py:606
            cx.set_item(
                &cx.get("dit_config")?,
                &text("axes_dims"),
                list(vec![
                    integer(32_i128)?,
                    integer(32_i128)?,
                    integer(32_i128)?,
                ]),
            )?;
            // comfy/model_detection.py:607
            cx.set_item(
                &cx.get("dit_config")?,
                &text("axes_lens"),
                list(vec![
                    integer(300_i128)?,
                    integer(512_i128)?,
                    integer(512_i128)?,
                ]),
            )?;
            // comfy/model_detection.py:608
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_theta"),
                float(10000.0_f64),
            )?;
            // comfy/model_detection.py:609
            cx.set_item(
                &cx.get("dit_config")?,
                &text("ffn_dim_multiplier"),
                float(4.0_f64),
            )?;
            // comfy/model_detection.py:610
            cx.set(
                "ctd_weight",
                cx.method(
                    &cx.get("state_dict")?,
                    "get",
                    vec![
                        cx.method(
                            &text("{}clip_text_pooled_proj.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        none(),
                    ],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:611
            if cx.truth(&{
                let detector_tmp_203 = cx.get("ctd_weight")?;
                let detector_tmp_204 = none();
                cx.compare("IsNot", &detector_tmp_203, &detector_tmp_204)?
            })? {
                // comfy/model_detection.py:612
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("clip_text_dim"),
                    cx.index(
                        &cx.attribute(&cx.get("ctd_weight")?, "shape")?,
                        &integer(0_i128)?,
                    )?,
                )?;
            }
        } else {
            // comfy/model_detection.py:614
            if cx.truth(&{
                let detector_tmp_205 = cx.index(&cx.get("dit_config")?, &text("dim"))?;
                let detector_tmp_206 = integer(3840_i128)?;
                cx.compare("Eq", &detector_tmp_205, &detector_tmp_206)?
            })? {
                // comfy/model_detection.py:615
                cx.set_item(&cx.get("dit_config")?, &text("n_heads"), integer(30_i128)?)?;
                // comfy/model_detection.py:616
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("n_kv_heads"),
                    integer(30_i128)?,
                )?;
                // comfy/model_detection.py:617
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("axes_dims"),
                    list(vec![
                        integer(32_i128)?,
                        integer(48_i128)?,
                        integer(48_i128)?,
                    ]),
                )?;
                // comfy/model_detection.py:618
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("axes_lens"),
                    list(vec![
                        integer(1536_i128)?,
                        integer(512_i128)?,
                        integer(512_i128)?,
                    ]),
                )?;
                // comfy/model_detection.py:619
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("rope_theta"),
                    float(256.0_f64),
                )?;
                // comfy/model_detection.py:620
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("ffn_dim_multiplier"),
                    cx.binary("Div", &float(8.0_f64), &float(3.0_f64))?,
                )?;
                // comfy/model_detection.py:621
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("z_image_modulation"),
                    boolean(true),
                )?;
                // comfy/model_detection.py:622
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("time_scale"),
                    float(1000.0_f64),
                )?;
                // comfy/model_detection.py:623
                cx.attempt(|cx| {
                    // comfy/model_detection.py:624
                    cx.set_item(&cx.get("dit_config")?, &text("allow_fp16"), {
                        let detector_tmp_207 = cx.method(
                            &cx.function(
                                "torch.std",
                                vec![cx.index(
                                    &cx.get("state_dict")?,
                                    &cx.method(
                                        &text("{}layers.{}.ffn_norm1.weight"),
                                        "format",
                                        vec![
                                            cx.get("key_prefix")?,
                                            cx.binary(
                                                "Sub",
                                                &cx.index(
                                                    &cx.get("dit_config")?,
                                                    &text("n_layers"),
                                                )?,
                                                &integer(2_i128)?,
                                            )?,
                                        ],
                                        vec![],
                                    )?,
                                )?],
                                vec![("unbiased", boolean(false))],
                            )?,
                            "item",
                            vec![],
                            vec![],
                        )?;
                        let detector_tmp_208 = float(0.42_f64);
                        cx.compare("Lt", &detector_tmp_207, &detector_tmp_208)?
                    })?;
                    Ok(())
                })?;
                // comfy/model_detection.py:627
                cx.set(
                    "dec_cond_key",
                    cx.method(
                        &text("{}dec_net.cond_embed.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                );
                // comfy/model_detection.py:628
                if cx.truth(&{
                    let detector_tmp_209 = cx.method(
                        &text("{}cap_pad_token"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_210 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_209, &detector_tmp_210)?
                })? {
                    // comfy/model_detection.py:629
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("pad_tokens_multiple"),
                        integer(32_i128)?,
                    )?;
                }
                // comfy/model_detection.py:631
                cx.set("ming_metadata", {
                    let detector_tmp_216 = {
                        let detector_tmp_217 = cx.get("metadata")?;
                        let detector_tmp_218 = none();
                        cx.compare("IsNot", &detector_tmp_217, &detector_tmp_218)?
                    };
                    if cx.truth(&detector_tmp_216)? {
                        {
                            let detector_tmp_213 = {
                                let detector_tmp_214 = text("config");
                                let detector_tmp_215 = cx.get("metadata")?;
                                cx.compare("In", &detector_tmp_214, &detector_tmp_215)?
                            };
                            if cx.truth(&detector_tmp_213)? {
                                {
                                    let detector_tmp_211 = cx.method(
                                        &cx.method(
                                            &cx.function(
                                                "json.loads",
                                                vec![cx.index(
                                                    &cx.get("metadata")?,
                                                    &text("config"),
                                                )?],
                                                vec![],
                                            )?,
                                            "get",
                                            vec![text("transformer"), cx.mapping(vec![])?],
                                            vec![],
                                        )?,
                                        "get",
                                        vec![text("image_model")],
                                        vec![],
                                    )?;
                                    let detector_tmp_212 = text("ming_image");
                                    cx.compare("Eq", &detector_tmp_211, &detector_tmp_212)?
                                }
                            } else {
                                detector_tmp_213
                            }
                        }
                    } else {
                        detector_tmp_216
                    }
                });
                // comfy/model_detection.py:632
                if cx.truth(&{
                    let detector_tmp_225 = {
                        let detector_tmp_226 = cx.method(
                            &text("{}__ming_image__"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?;
                        let detector_tmp_227 = cx.get("state_dict_keys")?;
                        cx.compare("In", &detector_tmp_226, &detector_tmp_227)?
                    };
                    if cx.truth(&detector_tmp_225)? {
                        detector_tmp_225
                    } else {
                        {
                            let detector_tmp_224 = cx.get("ming_metadata")?;
                            if cx.truth(&detector_tmp_224)? {
                                detector_tmp_224
                            } else {
                                {
                                    let detector_tmp_221 = {
                                        let detector_tmp_222 = text("pad_tokens_multiple");
                                        let detector_tmp_223 = cx.get("dit_config")?;
                                        cx.compare("NotIn", &detector_tmp_222, &detector_tmp_223)?
                                    };
                                    if cx.truth(&detector_tmp_221)? {
                                        {
                                            let detector_tmp_219 = cx.get("dec_cond_key")?;
                                            let detector_tmp_220 = cx.get("state_dict_keys")?;
                                            cx.compare(
                                                "NotIn",
                                                &detector_tmp_219,
                                                &detector_tmp_220,
                                            )?
                                        }
                                    } else {
                                        detector_tmp_221
                                    }
                                }
                            }
                        }
                    }
                })? {
                    // comfy/model_detection.py:633
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("image_model"),
                        text("ming_image"),
                    )?;
                    // comfy/model_detection.py:634
                    if cx.truth(&{
                        let detector_tmp_228 = text("pad_tokens_multiple");
                        let detector_tmp_229 = cx.get("dit_config")?;
                        cx.compare("NotIn", &detector_tmp_228, &detector_tmp_229)?
                    })? {
                        // comfy/model_detection.py:635
                        cx.set_item(
                            &cx.get("dit_config")?,
                            &text("masked_pad_multiple"),
                            integer(32_i128)?,
                        )?;
                    }
                }
                // comfy/model_detection.py:636
                cx.set(
                    "sig_weight",
                    cx.method(
                        &cx.get("state_dict")?,
                        "get",
                        vec![
                            cx.method(
                                &text("{}siglip_embedder.0.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                            none(),
                        ],
                        vec![],
                    )?,
                );
                // comfy/model_detection.py:637
                if cx.truth(&{
                    let detector_tmp_230 = cx.get("sig_weight")?;
                    let detector_tmp_231 = none();
                    cx.compare("IsNot", &detector_tmp_230, &detector_tmp_231)?
                })? {
                    // comfy/model_detection.py:638
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("siglip_feat_dim"),
                        cx.index(
                            &cx.attribute(&cx.get("sig_weight")?, "shape")?,
                            &integer(0_i128)?,
                        )?,
                    )?;
                }
                // comfy/model_detection.py:640
                if cx.truth(&{
                    let detector_tmp_232 = cx.get("dec_cond_key")?;
                    let detector_tmp_233 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_232, &detector_tmp_233)?
                })? {
                    // comfy/model_detection.py:641
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("image_model"),
                        text("zimage_pixel"),
                    )?;
                    // comfy/model_detection.py:645
                    cx.set(
                        "x_emb_in",
                        cx.index(
                            &cx.attribute(
                                &cx.index(
                                    &cx.get("state_dict")?,
                                    &cx.method(
                                        &text("{}x_embedder.weight"),
                                        "format",
                                        vec![cx.get("key_prefix")?],
                                        vec![],
                                    )?,
                                )?,
                                "shape",
                            )?,
                            &integer(1_i128)?,
                        )?,
                    );
                    // comfy/model_detection.py:646
                    cx.set(
                        "dec_out",
                        cx.index(
                            &cx.attribute(
                                &cx.index(
                                    &cx.get("state_dict")?,
                                    &cx.method(
                                        &text("{}dec_net.final_layer.linear.weight"),
                                        "format",
                                        vec![cx.get("key_prefix")?],
                                        vec![],
                                    )?,
                                )?,
                                "shape",
                            )?,
                            &integer(0_i128)?,
                        )?,
                    );
                    // comfy/model_detection.py:649
                    cx.set(
                        "embedder_w",
                        cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}dec_net.input_embedder.embedder.0.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                    );
                    // comfy/model_detection.py:650
                    cx.set("dec_in_ch", cx.get("dec_out")?);
                    // comfy/model_detection.py:651
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("patch_size"),
                        cx.function(
                            "round",
                            vec![cx.binary(
                                "Pow",
                                &cx.binary("Div", &cx.get("x_emb_in")?, &integer(3_i128)?)?,
                                &float(0.5_f64),
                            )?],
                            vec![],
                        )?,
                    )?;
                    // comfy/model_detection.py:652
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("in_channels"),
                        integer(3_i128)?,
                    )?;
                    // comfy/model_detection.py:653
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("decoder_in_channels"),
                        cx.get("dec_in_ch")?,
                    )?;
                    // comfy/model_detection.py:654
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("decoder_hidden_size"),
                        cx.index(
                            &cx.attribute(
                                &cx.index(&cx.get("state_dict")?, &cx.get("dec_cond_key")?)?,
                                "shape",
                            )?,
                            &integer(0_i128)?,
                        )?,
                    )?;
                    // comfy/model_detection.py:655
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("decoder_num_res_blocks"),
                        cx.function(
                            "count_blocks",
                            vec![
                                cx.get("state_dict_keys")?,
                                cx.binary(
                                    "Add",
                                    &cx.method(
                                        &text("{}dec_net.res_blocks."),
                                        "format",
                                        vec![cx.get("key_prefix")?],
                                        vec![],
                                    )?,
                                    &text("{}."),
                                )?,
                            ],
                            vec![],
                        )?,
                    )?;
                    // comfy/model_detection.py:658
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("decoder_max_freqs"),
                        cx.function(
                            "int",
                            vec![cx.binary(
                                "Pow",
                                &cx.binary(
                                    "Sub",
                                    &cx.index(
                                        &cx.attribute(&cx.get("embedder_w")?, "shape")?,
                                        &integer(1_i128)?,
                                    )?,
                                    &cx.get("dec_in_ch")?,
                                )?,
                                &float(0.5_f64),
                            )?],
                            vec![],
                        )?,
                    )?;
                    // comfy/model_detection.py:659
                    if cx.truth(&{
                        let detector_tmp_234 = cx.method(
                            &text("{}__x0__"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?;
                        let detector_tmp_235 = cx.get("state_dict_keys")?;
                        cx.compare("In", &detector_tmp_234, &detector_tmp_235)?
                    })? {
                        // comfy/model_detection.py:660
                        cx.set_item(&cx.get("dit_config")?, &text("use_x0"), boolean(true))?;
                    }
                }
            }
        }
        // comfy/model_detection.py:662
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:664
    if cx.truth(&{
        let detector_tmp_236 = cx.method(
            &text("{}blocks.0.norm1.linear.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_237 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_236, &detector_tmp_237)?
    })? {
        // comfy/model_detection.py:665
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:666
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("cogvideox"),
        )?;
        // comfy/model_detection.py:669
        cx.set(
            "norm1_weight",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}blocks.0.norm1.linear.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:670
        cx.set(
            "time_embed_dim",
            cx.index(
                &cx.attribute(&cx.get("norm1_weight")?, "shape")?,
                &integer(1_i128)?,
            )?,
        );
        // comfy/model_detection.py:671
        cx.set(
            "dim",
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(&cx.get("norm1_weight")?, "shape")?,
                    &integer(0_i128)?,
                )?,
                &integer(6_i128)?,
            )?,
        );
        // comfy/model_detection.py:673
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_attention_heads"),
            cx.binary("FloorDiv", &cx.get("dim")?, &integer(64_i128)?)?,
        )?;
        // comfy/model_detection.py:674
        cx.set_item(
            &cx.get("dit_config")?,
            &text("attention_head_dim"),
            integer(64_i128)?,
        )?;
        // comfy/model_detection.py:675
        cx.set_item(
            &cx.get("dit_config")?,
            &text("time_embed_dim"),
            cx.get("time_embed_dim")?,
        )?;
        // comfy/model_detection.py:676
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:679
        cx.set(
            "patch_proj_key",
            cx.method(
                &text("{}patch_embed.proj.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:680
        if cx.truth(&{
            let detector_tmp_238 = cx.get("patch_proj_key")?;
            let detector_tmp_239 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_238, &detector_tmp_239)?
        })? {
            // comfy/model_detection.py:681
            cx.set(
                "w",
                cx.index(&cx.get("state_dict")?, &cx.get("patch_proj_key")?)?,
            );
            // comfy/model_detection.py:682
            if cx.truth(&{
                let detector_tmp_240 = cx.attribute(&cx.get("w")?, "ndim")?;
                let detector_tmp_241 = integer(4_i128)?;
                cx.compare("Eq", &detector_tmp_240, &detector_tmp_241)?
            })? {
                // comfy/model_detection.py:684
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("in_channels"),
                    cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(1_i128)?)?,
                )?;
                // comfy/model_detection.py:685
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("patch_size"),
                    cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(2_i128)?)?,
                )?;
            } else {
                // comfy/model_detection.py:686
                if cx.truth(&{
                    let detector_tmp_242 = cx.attribute(&cx.get("w")?, "ndim")?;
                    let detector_tmp_243 = integer(2_i128)?;
                    cx.compare("Eq", &detector_tmp_242, &detector_tmp_243)?
                })? {
                    // comfy/model_detection.py:688
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("patch_size"),
                        integer(2_i128)?,
                    )?;
                    // comfy/model_detection.py:689
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("patch_size_t"),
                        integer(2_i128)?,
                    )?;
                    // comfy/model_detection.py:690
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("in_channels"),
                        cx.binary(
                            "FloorDiv",
                            &cx.index(&cx.attribute(&cx.get("w")?, "shape")?, &integer(1_i128)?)?,
                            &cx.binary(
                                "Mult",
                                &cx.binary("Mult", &integer(2_i128)?, &integer(2_i128)?)?,
                                &integer(2_i128)?,
                            )?,
                        )?,
                    )?;
                }
            }
        }
        // comfy/model_detection.py:692
        cx.set(
            "text_proj_key",
            cx.method(
                &text("{}patch_embed.text_proj.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:693
        if cx.truth(&{
            let detector_tmp_244 = cx.get("text_proj_key")?;
            let detector_tmp_245 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_244, &detector_tmp_245)?
        })? {
            // comfy/model_detection.py:694
            cx.set_item(
                &cx.get("dit_config")?,
                &text("text_embed_dim"),
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("text_proj_key")?)?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:697
        cx.set(
            "ofs_key",
            cx.method(
                &text("{}ofs_embedding_linear_1.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:698
        if cx.truth(&{
            let detector_tmp_246 = cx.get("ofs_key")?;
            let detector_tmp_247 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_246, &detector_tmp_247)?
        })? {
            // comfy/model_detection.py:699
            cx.set_item(
                &cx.get("dit_config")?,
                &text("ofs_embed_dim"),
                cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("ofs_key")?)?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:702
        cx.set(
            "pos_key",
            cx.method(
                &text("{}patch_embed.pos_embedding"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:703
        if cx.truth(&{
            let detector_tmp_248 = cx.get("pos_key")?;
            let detector_tmp_249 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_248, &detector_tmp_249)?
        })? {
            // comfy/model_detection.py:704
            cx.set_item(
                &cx.get("dit_config")?,
                &text("use_learned_positional_embeddings"),
                boolean(true),
            )?;
            // comfy/model_detection.py:705
            cx.set_item(
                &cx.get("dit_config")?,
                &text("use_rotary_positional_embeddings"),
                boolean(false),
            )?;
        } else {
            // comfy/model_detection.py:707
            cx.set_item(
                &cx.get("dit_config")?,
                &text("use_learned_positional_embeddings"),
                boolean(false),
            )?;
            // comfy/model_detection.py:708
            cx.set_item(
                &cx.get("dit_config")?,
                &text("use_rotary_positional_embeddings"),
                boolean(true),
            )?;
        }
        // comfy/model_detection.py:710
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:712
    cx.set(
        "seedvr2_7b_separate_key",
        cx.method(
            &text("{}blocks.35.mlp.vid.proj_out.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:713
    if cx.truth(&{
        let detector_tmp_252 = {
            let detector_tmp_253 = cx.get("seedvr2_7b_separate_key")?;
            let detector_tmp_254 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_253, &detector_tmp_254)?
        };
        if cx.truth(&detector_tmp_252)? {
            {
                let detector_tmp_250 = cx.index(
                    &cx.attribute(
                        &cx.index(&cx.get("state_dict")?, &cx.get("seedvr2_7b_separate_key")?)?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?;
                let detector_tmp_251 = integer(3072_i128)?;
                cx.compare("Eq", &detector_tmp_250, &detector_tmp_251)?
            }
        } else {
            detector_tmp_252
        }
    })? {
        // comfy/model_detection.py:714
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:715
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("seedvr2"),
        )?;
        // comfy/model_detection.py:716
        cx.set_item(
            &cx.get("dit_config")?,
            &text("vid_dim"),
            integer(3072_i128)?,
        )?;
        // comfy/model_detection.py:717
        cx.set_item(&cx.get("dit_config")?, &text("heads"), integer(24_i128)?)?;
        // comfy/model_detection.py:718
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            integer(36_i128)?,
        )?;
        // comfy/model_detection.py:720
        cx.set_item(
            &cx.get("dit_config")?,
            &text("mm_layers"),
            integer(36_i128)?,
        )?;
        // comfy/model_detection.py:721
        cx.set_item(&cx.get("dit_config")?, &text("norm_eps"), float(1e-05_f64))?;
        // comfy/model_detection.py:722
        cx.set_item(&cx.get("dit_config")?, &text("rope_type"), text("rope3d"))?;
        // comfy/model_detection.py:723
        cx.set_item(&cx.get("dit_config")?, &text("rope_dim"), integer(64_i128)?)?;
        // comfy/model_detection.py:724
        cx.set_item(&cx.get("dit_config")?, &text("mlp_type"), text("normal"))?;
        // comfy/model_detection.py:725
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:726
    if cx.truth(&{
        let detector_tmp_255 = cx.method(
            &text("{}blocks.35.mlp.all.proj_in_gate.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_256 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_255, &detector_tmp_256)?
    })? {
        // comfy/model_detection.py:727
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:728
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("seedvr2"),
        )?;
        // comfy/model_detection.py:729
        cx.set_item(
            &cx.get("dit_config")?,
            &text("vid_dim"),
            integer(3072_i128)?,
        )?;
        // comfy/model_detection.py:730
        cx.set_item(&cx.get("dit_config")?, &text("heads"), integer(24_i128)?)?;
        // comfy/model_detection.py:731
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            integer(36_i128)?,
        )?;
        // comfy/model_detection.py:733
        cx.set_item(
            &cx.get("dit_config")?,
            &text("mm_layers"),
            integer(10_i128)?,
        )?;
        // comfy/model_detection.py:734
        cx.set_item(&cx.get("dit_config")?, &text("norm_eps"), float(1e-05_f64))?;
        // comfy/model_detection.py:735
        cx.set_item(&cx.get("dit_config")?, &text("rope_type"), text("rope3d"))?;
        // comfy/model_detection.py:736
        cx.set_item(&cx.get("dit_config")?, &text("rope_dim"), integer(64_i128)?)?;
        // comfy/model_detection.py:737
        cx.set_item(&cx.get("dit_config")?, &text("mlp_type"), text("swiglu"))?;
        // comfy/model_detection.py:738
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:739
    if cx.truth(&{
        let detector_tmp_257 = cx.method(
            &text("{}blocks.31.mlp.all.proj_in_gate.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_258 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_257, &detector_tmp_258)?
    })? {
        // comfy/model_detection.py:740
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:741
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("seedvr2"),
        )?;
        // comfy/model_detection.py:742
        cx.set_item(
            &cx.get("dit_config")?,
            &text("vid_dim"),
            integer(2560_i128)?,
        )?;
        // comfy/model_detection.py:743
        cx.set_item(&cx.get("dit_config")?, &text("heads"), integer(20_i128)?)?;
        // comfy/model_detection.py:744
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            integer(32_i128)?,
        )?;
        // comfy/model_detection.py:745
        cx.set_item(&cx.get("dit_config")?, &text("norm_eps"), float(1e-05_f64))?;
        // comfy/model_detection.py:746
        cx.set_item(&cx.get("dit_config")?, &text("mlp_type"), text("swiglu"))?;
        // comfy/model_detection.py:747
        cx.set_item(&cx.get("dit_config")?, &text("vid_out_norm"), boolean(true))?;
        // comfy/model_detection.py:748
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:750
    if cx.truth(&{
        let detector_tmp_259 = cx.method(
            &text("{}head.modulation"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_260 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_259, &detector_tmp_260)?
    })? {
        // comfy/model_detection.py:751
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:752
        cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("wan2.1"))?;
        // comfy/model_detection.py:753
        cx.set(
            "dim",
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}head.modulation"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &cx.unary("USub", &integer(1_i128)?)?,
            )?,
        );
        // comfy/model_detection.py:754
        cx.set(
            "out_dim",
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}head.head.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &integer(4_i128)?,
            )?,
        );
        // comfy/model_detection.py:755
        cx.set_item(&cx.get("dit_config")?, &text("dim"), cx.get("dim")?)?;
        // comfy/model_detection.py:756
        cx.set_item(&cx.get("dit_config")?, &text("out_dim"), cx.get("out_dim")?)?;
        // comfy/model_detection.py:757
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            cx.binary("FloorDiv", &cx.get("dim")?, &integer(128_i128)?)?,
        )?;
        // comfy/model_detection.py:758
        cx.set_item(
            &cx.get("dit_config")?,
            &text("ffn_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}blocks.0.ffn.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:759
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:760
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            tuple(vec![integer(1_i128)?, integer(2_i128)?, integer(2_i128)?]),
        )?;
        // comfy/model_detection.py:761
        cx.set_item(
            &cx.get("dit_config")?,
            &text("freq_dim"),
            integer(256_i128)?,
        )?;
        // comfy/model_detection.py:762
        cx.set_item(
            &cx.get("dit_config")?,
            &text("window_size"),
            tuple(vec![
                cx.unary("USub", &integer(1_i128)?)?,
                cx.unary("USub", &integer(1_i128)?)?,
            ]),
        )?;
        // comfy/model_detection.py:763
        cx.set_item(&cx.get("dit_config")?, &text("qk_norm"), boolean(true))?;
        // comfy/model_detection.py:764
        cx.set_item(
            &cx.get("dit_config")?,
            &text("cross_attn_norm"),
            boolean(true),
        )?;
        // comfy/model_detection.py:765
        cx.set_item(&cx.get("dit_config")?, &text("eps"), float(1e-06_f64))?;
        // comfy/model_detection.py:766
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}patch_embedding.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:767
        if cx.truth(&{
            let detector_tmp_261 = cx.method(
                &text("{}vace_patch_embedding.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_262 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_261, &detector_tmp_262)?
        })? {
            // comfy/model_detection.py:768
            cx.set_item(&cx.get("dit_config")?, &text("model_type"), text("vace"))?;
            // comfy/model_detection.py:769
            cx.set_item(
                &cx.get("dit_config")?,
                &text("vace_in_dim"),
                cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}vace_patch_embedding.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            )?;
            // comfy/model_detection.py:770
            cx.set_item(
                &cx.get("dit_config")?,
                &text("vace_layers"),
                cx.function(
                    "count_blocks",
                    vec![
                        cx.get("state_dict_keys")?,
                        cx.binary(
                            "Add",
                            &cx.method(
                                &text("{}vace_blocks."),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                            &text("{}."),
                        )?,
                    ],
                    vec![],
                )?,
            )?;
            // comfy/model_detection.py:771
            if cx.truth(&{
                let detector_tmp_263 = cx.method(
                    &text("{}img_emb.proj.0.bias"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_264 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_263, &detector_tmp_264)?
            })? {
                // comfy/model_detection.py:772
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("vace_image_input"),
                    boolean(true),
                )?;
            }
        } else {
            // comfy/model_detection.py:773
            if cx.truth(&{
                let detector_tmp_265 = cx.method(
                    &text("{}control_adapter.conv.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_266 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_265, &detector_tmp_266)?
            })? {
                // comfy/model_detection.py:774
                if cx.truth(&{
                    let detector_tmp_267 = cx.method(
                        &text("{}img_emb.proj.0.bias"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_268 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_267, &detector_tmp_268)?
                })? {
                    // comfy/model_detection.py:775
                    cx.set_item(&cx.get("dit_config")?, &text("model_type"), text("camera"))?;
                } else {
                    // comfy/model_detection.py:777
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("model_type"),
                        text("camera_2.2"),
                    )?;
                }
            } else {
                // comfy/model_detection.py:778
                if cx.truth(&{
                    let detector_tmp_269 = cx.method(
                        &text("{}casual_audio_encoder.encoder.final_linear.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_270 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_269, &detector_tmp_270)?
                })? {
                    // comfy/model_detection.py:779
                    cx.set_item(&cx.get("dit_config")?, &text("model_type"), text("s2v"))?;
                } else {
                    // comfy/model_detection.py:780
                    if cx.truth(&{
                        let detector_tmp_271 = cx.method(
                            &text("{}audio_proj.audio_proj_glob_1.layer.bias"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?;
                        let detector_tmp_272 = cx.get("state_dict_keys")?;
                        cx.compare("In", &detector_tmp_271, &detector_tmp_272)?
                    })? {
                        // comfy/model_detection.py:781
                        cx.set_item(&cx.get("dit_config")?, &text("model_type"), text("humo"))?;
                    } else {
                        // comfy/model_detection.py:782
                        if cx.truth(&{
                            let detector_tmp_273 = cx.method(
                                &text("{}face_adapter.fuser_blocks.0.k_norm.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?;
                            let detector_tmp_274 = cx.get("state_dict_keys")?;
                            cx.compare("In", &detector_tmp_273, &detector_tmp_274)?
                        })? {
                            // comfy/model_detection.py:783
                            cx.set_item(
                                &cx.get("dit_config")?,
                                &text("model_type"),
                                text("animate"),
                            )?;
                        } else {
                            // comfy/model_detection.py:784
                            if cx.truth(&{
                                let detector_tmp_275 = cx.method(
                                    &text("{}patch_embedding_mask.weight"),
                                    "format",
                                    vec![cx.get("key_prefix")?],
                                    vec![],
                                )?;
                                let detector_tmp_276 = cx.get("state_dict_keys")?;
                                cx.compare("In", &detector_tmp_275, &detector_tmp_276)?
                            })? {
                                // comfy/model_detection.py:785
                                cx.set_item(
                                    &cx.get("dit_config")?,
                                    &text("model_type"),
                                    text("scail2"),
                                )?;
                            } else {
                                // comfy/model_detection.py:786
                                if cx.truth(&{
                                    let detector_tmp_277 = cx.method(
                                        &text("{}patch_embedding_pose.weight"),
                                        "format",
                                        vec![cx.get("key_prefix")?],
                                        vec![],
                                    )?;
                                    let detector_tmp_278 = cx.get("state_dict_keys")?;
                                    cx.compare("In", &detector_tmp_277, &detector_tmp_278)?
                                })? {
                                    // comfy/model_detection.py:787
                                    cx.set_item(
                                        &cx.get("dit_config")?,
                                        &text("model_type"),
                                        text("scail"),
                                    )?;
                                } else {
                                    // comfy/model_detection.py:788
                                    if cx.truth(&{
                                        let detector_tmp_279 = cx.method(
                                            &text("{}patch_embedding_global.weight"),
                                            "format",
                                            vec![cx.get("key_prefix")?],
                                            vec![],
                                        )?;
                                        let detector_tmp_280 = cx.get("state_dict_keys")?;
                                        cx.compare("In", &detector_tmp_279, &detector_tmp_280)?
                                    })? {
                                        // comfy/model_detection.py:789
                                        cx.set_item(
                                            &cx.get("dit_config")?,
                                            &text("model_type"),
                                            text("wandancer"),
                                        )?;
                                    } else {
                                        // comfy/model_detection.py:791
                                        if cx.truth(&{
                                            let detector_tmp_281 = cx.method(
                                                &text("{}img_emb.proj.0.bias"),
                                                "format",
                                                vec![cx.get("key_prefix")?],
                                                vec![],
                                            )?;
                                            let detector_tmp_282 = cx.get("state_dict_keys")?;
                                            cx.compare("In", &detector_tmp_281, &detector_tmp_282)?
                                        })? {
                                            // comfy/model_detection.py:792
                                            cx.set_item(
                                                &cx.get("dit_config")?,
                                                &text("model_type"),
                                                text("i2v"),
                                            )?;
                                        } else {
                                            // comfy/model_detection.py:794
                                            cx.set_item(
                                                &cx.get("dit_config")?,
                                                &text("model_type"),
                                                text("t2v"),
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // comfy/model_detection.py:795
        cx.set(
            "flf_weight",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![cx.method(
                    &text("{}img_emb.emb_pos"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:796
        if cx.truth(&{
            let detector_tmp_283 = cx.get("flf_weight")?;
            let detector_tmp_284 = none();
            cx.compare("IsNot", &detector_tmp_283, &detector_tmp_284)?
        })? {
            // comfy/model_detection.py:797
            cx.set_item(
                &cx.get("dit_config")?,
                &text("flf_pos_embed_token_number"),
                cx.index(
                    &cx.attribute(&cx.get("flf_weight")?, "shape")?,
                    &integer(1_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:799
        cx.set(
            "ref_conv_weight",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![cx.method(
                    &text("{}ref_conv.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:800
        if cx.truth(&{
            let detector_tmp_285 = cx.get("ref_conv_weight")?;
            let detector_tmp_286 = none();
            cx.compare("IsNot", &detector_tmp_285, &detector_tmp_286)?
        })? {
            // comfy/model_detection.py:801
            cx.set_item(
                &cx.get("dit_config")?,
                &text("in_dim_ref_conv"),
                cx.index(
                    &cx.attribute(&cx.get("ref_conv_weight")?, "shape")?,
                    &integer(1_i128)?,
                )?,
            )?;
        }
        // comfy/model_detection.py:803
        if cx.truth(&{
            let detector_tmp_289 = {
                let detector_tmp_290 = cx.get("metadata")?;
                let detector_tmp_291 = none();
                cx.compare("IsNot", &detector_tmp_290, &detector_tmp_291)?
            };
            if cx.truth(&detector_tmp_289)? {
                {
                    let detector_tmp_287 = text("config");
                    let detector_tmp_288 = cx.get("metadata")?;
                    cx.compare("In", &detector_tmp_287, &detector_tmp_288)?
                }
            } else {
                detector_tmp_289
            }
        })? {
            // comfy/model_detection.py:804
            cx.method(
                &cx.get("dit_config")?,
                "update",
                vec![cx.method(
                    &cx.function(
                        "json.loads",
                        vec![cx.index(&cx.get("metadata")?, &text("config"))?],
                        vec![],
                    )?,
                    "get",
                    vec![text("transformer"), cx.mapping(vec![])?],
                    vec![],
                )?],
                vec![],
            )?;
        }
        // comfy/model_detection.py:806
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:808
    if cx.truth(&{
        let detector_tmp_292 = cx.method(
            &text("{}latent_in.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_293 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_292, &detector_tmp_293)?
    })? {
        // comfy/model_detection.py:809
        cx.set(
            "in_shape",
            cx.attribute(
                &cx.index(
                    &cx.get("state_dict")?,
                    &cx.method(
                        &text("{}latent_in.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                )?,
                "shape",
            )?,
        );
        // comfy/model_detection.py:810
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:811
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("hunyuan3d2"),
        )?;
        // comfy/model_detection.py:812
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.index(&cx.get("in_shape")?, &integer(1_i128)?)?,
        )?;
        // comfy/model_detection.py:813
        cx.set_item(
            &cx.get("dit_config")?,
            &text("context_in_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}cond_in.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:814
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            cx.index(&cx.get("in_shape")?, &integer(0_i128)?)?,
        )?;
        // comfy/model_detection.py:815
        cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio"), float(4.0_f64))?;
        // comfy/model_detection.py:816
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:817
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}double_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:818
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth_single_blocks"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}single_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:819
        cx.set_item(&cx.get("dit_config")?, &text("qkv_bias"), boolean(true))?;
        // comfy/model_detection.py:820
        cx.set_item(&cx.get("dit_config")?, &text("guidance_embed"), {
            let detector_tmp_294 = cx.method(
                &text("{}guidance_in.in_layer.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_295 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_294, &detector_tmp_295)?
        })?;
        // comfy/model_detection.py:821
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:823
    if cx.truth(&{
        let detector_tmp_298 = {
            let detector_tmp_299 = cx.concatenate_text(&[
                cx.string(&cx.get("key_prefix")?)?,
                text("t_embedder.mlp.2.weight"),
            ])?;
            let detector_tmp_300 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_299, &detector_tmp_300)?
        };
        if cx.truth(&detector_tmp_298)? {
            {
                let detector_tmp_296 = cx.concatenate_text(&[
                    cx.string(&cx.get("key_prefix")?)?,
                    text("blocks.0.attn1.k_norm.weight"),
                ])?;
                let detector_tmp_297 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_296, &detector_tmp_297)?
            }
        } else {
            detector_tmp_298
        }
    })? {
        // comfy/model_detection.py:824
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:825
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("hunyuan3d2_1"),
        )?;
        // comfy/model_detection.py:826
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.concatenate_text(&[
                            cx.string(&cx.get("key_prefix")?)?,
                            text("x_embedder.weight"),
                        ])?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:827
        cx.set_item(
            &cx.get("dit_config")?,
            &text("context_dim"),
            integer(1024_i128)?,
        )?;
        // comfy/model_detection.py:828
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.concatenate_text(&[
                            cx.string(&cx.get("key_prefix")?)?,
                            text("x_embedder.weight"),
                        ])?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:829
        cx.set_item(&cx.get("dit_config")?, &text("mlp_ratio"), float(4.0_f64))?;
        // comfy/model_detection.py:830
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:831
        cx.set_item(
            &cx.get("dit_config")?,
            &text("depth"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.concatenate_text(&[cx.string(&cx.get("key_prefix")?)?, text("blocks.{}")])?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:832
        cx.set_item(&cx.get("dit_config")?, &text("qkv_bias"), boolean(false))?;
        // comfy/model_detection.py:833
        cx.set_item(
            &cx.get("dit_config")?,
            &text("guidance_cond_proj_dim"),
            none(),
        )?;
        // comfy/model_detection.py:834
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:836
    if cx.truth(&{
        let detector_tmp_303 = {
            let detector_tmp_304 = cx.method(
                &text("{}cam_out_layer.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_305 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_304, &detector_tmp_305)?
        };
        if cx.truth(&detector_tmp_303)? {
            {
                let detector_tmp_301 = cx.method(
                    &text("{}repo_layers.0.final_map.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_302 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_301, &detector_tmp_302)?
            }
        } else {
            detector_tmp_303
        }
    })? {
        // comfy/model_detection.py:837
        return cx.mapping(vec![(text("image_model"), text("triposplat"))]);
    }
    // comfy/model_detection.py:839
    if cx.truth(&{
        let detector_tmp_308 = {
            let detector_tmp_309 = cx.method(
                &text("{}t_embedder1.mlp.0.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_310 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_309, &detector_tmp_310)?
        };
        if cx.truth(&detector_tmp_308)? {
            {
                let detector_tmp_306 = cx.method(
                    &text("{}x_embedder.proj1.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_307 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_306, &detector_tmp_307)?
            }
        } else {
            detector_tmp_308
        }
    })? {
        // comfy/model_detection.py:840
        return cx.mapping(vec![(text("image_model"), text("hidream_o1"))]);
    }
    // comfy/model_detection.py:842
    cx.set(
        "vision_key",
        cx.concatenate_text(&[
            cx.string(&cx.get("key_prefix")?)?,
            text("fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"),
        ])?,
    );
    // comfy/model_detection.py:843
    cx.set(
        "query_key",
        cx.concatenate_text(&[
            cx.string(&cx.get("key_prefix")?)?,
            text("language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"),
        ])?,
    );
    // comfy/model_detection.py:844
    if cx.truth(&{
        let detector_tmp_319 = {
            let detector_tmp_320 = cx.get("vision_key")?;
            let detector_tmp_321 = cx.get("state_dict")?;
            cx.compare("In", &detector_tmp_320, &detector_tmp_321)?
        };
        if cx.truth(&detector_tmp_319)? {
            {
                let detector_tmp_316 = {
                    let detector_tmp_317 = cx.get("query_key")?;
                    let detector_tmp_318 = cx.get("state_dict")?;
                    cx.compare("In", &detector_tmp_317, &detector_tmp_318)?
                };
                if cx.truth(&detector_tmp_316)? {
                    {
                        let detector_tmp_313 = {
                            let detector_tmp_314 = cx.index(
                                &cx.attribute(
                                    &cx.index(&cx.get("state_dict")?, &cx.get("vision_key")?)?,
                                    "shape",
                                )?,
                                &integer(0_i128)?,
                            )?;
                            let detector_tmp_315 = integer(1024_i128)?;
                            cx.compare("Eq", &detector_tmp_314, &detector_tmp_315)?
                        };
                        if cx.truth(&detector_tmp_313)? {
                            {
                                let detector_tmp_311 = cx.index(
                                    &cx.attribute(
                                        &cx.index(&cx.get("state_dict")?, &cx.get("query_key")?)?,
                                        "shape",
                                    )?,
                                    &integer(0_i128)?,
                                )?;
                                let detector_tmp_312 = integer(4096_i128)?;
                                cx.compare("Eq", &detector_tmp_311, &detector_tmp_312)?
                            }
                        } else {
                            detector_tmp_313
                        }
                    }
                } else {
                    detector_tmp_316
                }
            }
        } else {
            detector_tmp_319
        }
    })? {
        // comfy/model_detection.py:850
        return cx.mapping(vec![(text("image_model"), text("sensenova_u15"))]);
    }
    // comfy/model_detection.py:852
    if cx.truth(&{
        let detector_tmp_322 = cx.method(
            &text("{}caption_projection.0.linear.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_323 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_322, &detector_tmp_323)?
    })? {
        // comfy/model_detection.py:853
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:854
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("hidream"),
        )?;
        // comfy/model_detection.py:855
        cx.set_item(
            &cx.get("dit_config")?,
            &text("attention_head_dim"),
            integer(128_i128)?,
        )?;
        // comfy/model_detection.py:856
        cx.set_item(
            &cx.get("dit_config")?,
            &text("axes_dims_rope"),
            list(vec![
                integer(64_i128)?,
                integer(32_i128)?,
                integer(32_i128)?,
            ]),
        )?;
        // comfy/model_detection.py:857
        cx.set_item(
            &cx.get("dit_config")?,
            &text("caption_channels"),
            list(vec![integer(4096_i128)?, integer(4096_i128)?]),
        )?;
        // comfy/model_detection.py:858
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_resolution"),
            list(vec![integer(128_i128)?, integer(128_i128)?]),
        )?;
        // comfy/model_detection.py:859
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:860
        cx.set_item(
            &cx.get("dit_config")?,
            &text("llama_layers"),
            list(vec![
                integer(0_i128)?,
                integer(1_i128)?,
                integer(2_i128)?,
                integer(3_i128)?,
                integer(4_i128)?,
                integer(5_i128)?,
                integer(6_i128)?,
                integer(7_i128)?,
                integer(8_i128)?,
                integer(9_i128)?,
                integer(10_i128)?,
                integer(11_i128)?,
                integer(12_i128)?,
                integer(13_i128)?,
                integer(14_i128)?,
                integer(15_i128)?,
                integer(16_i128)?,
                integer(17_i128)?,
                integer(18_i128)?,
                integer(19_i128)?,
                integer(20_i128)?,
                integer(21_i128)?,
                integer(22_i128)?,
                integer(23_i128)?,
                integer(24_i128)?,
                integer(25_i128)?,
                integer(26_i128)?,
                integer(27_i128)?,
                integer(28_i128)?,
                integer(29_i128)?,
                integer(30_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
                integer(31_i128)?,
            ]),
        )?;
        // comfy/model_detection.py:861
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_attention_heads"),
            integer(20_i128)?,
        )?;
        // comfy/model_detection.py:862
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_routed_experts"),
            integer(4_i128)?,
        )?;
        // comfy/model_detection.py:863
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_activated_experts"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:864
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:865
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_single_layers"),
            integer(32_i128)?,
        )?;
        // comfy/model_detection.py:866
        cx.set_item(
            &cx.get("dit_config")?,
            &text("out_channels"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:867
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:868
        cx.set_item(
            &cx.get("dit_config")?,
            &text("text_emb_dim"),
            integer(2048_i128)?,
        )?;
        // comfy/model_detection.py:869
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:871
    if cx.truth(&{
        let detector_tmp_324 = cx.method(
            &text("{}blocks.0.mlp.layer1.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_325 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_324, &detector_tmp_325)?
    })? {
        // comfy/model_detection.py:872
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:873
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("cosmos_predict2"),
        )?;
        // comfy/model_detection.py:874
        if cx.truth(&{
            let detector_tmp_326 = cx.method(
                &text("{}llm_adapter.blocks.0.cross_attn.q_proj.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_327 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_326, &detector_tmp_327)?
        })? {
            // comfy/model_detection.py:875
            cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("anima"))?;
        }
        // comfy/model_detection.py:876
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_img_h"),
            integer(240_i128)?,
        )?;
        // comfy/model_detection.py:877
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_img_w"),
            integer(240_i128)?,
        )?;
        // comfy/model_detection.py:878
        cx.set_item(
            &cx.get("dit_config")?,
            &text("max_frames"),
            integer(128_i128)?,
        )?;
        // comfy/model_detection.py:879
        cx.set("concat_padding_mask", boolean(true));
        // comfy/model_detection.py:880
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.binary(
                "Sub",
                &cx.binary(
                    "FloorDiv",
                    &cx.index(
                        &cx.attribute(
                            &cx.index(
                                &cx.get("state_dict")?,
                                &cx.method(
                                    &text("{}x_embedder.proj.1.weight"),
                                    "format",
                                    vec![cx.get("key_prefix")?],
                                    vec![],
                                )?,
                            )?,
                            "shape",
                        )?,
                        &integer(1_i128)?,
                    )?,
                    &integer(4_i128)?,
                )?,
                &cx.function("int", vec![cx.get("concat_padding_mask")?], vec![])?,
            )?,
        )?;
        // comfy/model_detection.py:881
        cx.set_item(
            &cx.get("dit_config")?,
            &text("out_channels"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:882
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_spatial"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:883
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_temporal"),
            integer(1_i128)?,
        )?;
        // comfy/model_detection.py:884
        cx.set_item(
            &cx.get("dit_config")?,
            &text("model_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}x_embedder.proj.1.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:885
        cx.set_item(
            &cx.get("dit_config")?,
            &text("concat_padding_mask"),
            cx.get("concat_padding_mask")?,
        )?;
        // comfy/model_detection.py:886
        cx.set_item(
            &cx.get("dit_config")?,
            &text("crossattn_emb_channels"),
            integer(1024_i128)?,
        )?;
        // comfy/model_detection.py:887
        cx.set_item(&cx.get("dit_config")?, &text("pos_emb_cls"), text("rope3d"))?;
        // comfy/model_detection.py:888
        cx.set_item(
            &cx.get("dit_config")?,
            &text("pos_emb_learnable"),
            boolean(true),
        )?;
        // comfy/model_detection.py:889
        cx.set_item(
            &cx.get("dit_config")?,
            &text("pos_emb_interpolation"),
            text("crop"),
        )?;
        // comfy/model_detection.py:890
        cx.set_item(&cx.get("dit_config")?, &text("min_fps"), integer(1_i128)?)?;
        // comfy/model_detection.py:891
        cx.set_item(&cx.get("dit_config")?, &text("max_fps"), integer(30_i128)?)?;
        // comfy/model_detection.py:893
        cx.set_item(
            &cx.get("dit_config")?,
            &text("use_adaln_lora"),
            boolean(true),
        )?;
        // comfy/model_detection.py:894
        cx.set_item(
            &cx.get("dit_config")?,
            &text("adaln_lora_dim"),
            integer(256_i128)?,
        )?;
        // comfy/model_detection.py:895
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_blocks"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:896
        if cx.truth(&{
            let detector_tmp_328 = cx.index(&cx.get("dit_config")?, &text("model_channels"))?;
            let detector_tmp_329 = integer(2048_i128)?;
            cx.compare("Eq", &detector_tmp_328, &detector_tmp_329)?
        })? {
            // comfy/model_detection.py:897
            cx.set_item(
                &cx.get("dit_config")?,
                &text("num_heads"),
                integer(16_i128)?,
            )?;
        } else {
            // comfy/model_detection.py:898
            if cx.truth(&{
                let detector_tmp_330 = cx.index(&cx.get("dit_config")?, &text("model_channels"))?;
                let detector_tmp_331 = integer(5120_i128)?;
                cx.compare("Eq", &detector_tmp_330, &detector_tmp_331)?
            })? {
                // comfy/model_detection.py:899
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("num_heads"),
                    integer(40_i128)?,
                )?;
            }
        }
        // comfy/model_detection.py:901
        if cx.truth(&{
            let detector_tmp_332 = cx.index(&cx.get("dit_config")?, &text("in_channels"))?;
            let detector_tmp_333 = integer(16_i128)?;
            cx.compare("Eq", &detector_tmp_332, &detector_tmp_333)?
        })? {
            // comfy/model_detection.py:902
            cx.set_item(
                &cx.get("dit_config")?,
                &text("extra_per_block_abs_pos_emb"),
                boolean(false),
            )?;
            // comfy/model_detection.py:903
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_h_extrapolation_ratio"),
                float(4.0_f64),
            )?;
            // comfy/model_detection.py:904
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_w_extrapolation_ratio"),
                float(4.0_f64),
            )?;
            // comfy/model_detection.py:905
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_t_extrapolation_ratio"),
                float(1.0_f64),
            )?;
        } else {
            // comfy/model_detection.py:906
            if cx.truth(&{
                let detector_tmp_334 = cx.index(&cx.get("dit_config")?, &text("in_channels"))?;
                let detector_tmp_335 = integer(17_i128)?;
                cx.compare("Eq", &detector_tmp_334, &detector_tmp_335)?
            })? {
                // comfy/model_detection.py:907
                if cx.truth(&{
                    let detector_tmp_336 =
                        cx.index(&cx.get("dit_config")?, &text("model_channels"))?;
                    let detector_tmp_337 = integer(2048_i128)?;
                    cx.compare("Eq", &detector_tmp_336, &detector_tmp_337)?
                })? {
                    // comfy/model_detection.py:908
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("extra_per_block_abs_pos_emb"),
                        boolean(false),
                    )?;
                    // comfy/model_detection.py:909
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("rope_h_extrapolation_ratio"),
                        float(3.0_f64),
                    )?;
                    // comfy/model_detection.py:910
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("rope_w_extrapolation_ratio"),
                        float(3.0_f64),
                    )?;
                    // comfy/model_detection.py:911
                    cx.set_item(
                        &cx.get("dit_config")?,
                        &text("rope_t_extrapolation_ratio"),
                        float(1.0_f64),
                    )?;
                } else {
                    // comfy/model_detection.py:912
                    if cx.truth(&{
                        let detector_tmp_338 =
                            cx.index(&cx.get("dit_config")?, &text("model_channels"))?;
                        let detector_tmp_339 = integer(5120_i128)?;
                        cx.compare("Eq", &detector_tmp_338, &detector_tmp_339)?
                    })? {
                        // comfy/model_detection.py:913
                        cx.set_item(
                            &cx.get("dit_config")?,
                            &text("rope_h_extrapolation_ratio"),
                            float(2.0_f64),
                        )?;
                        // comfy/model_detection.py:914
                        cx.set_item(
                            &cx.get("dit_config")?,
                            &text("rope_w_extrapolation_ratio"),
                            float(2.0_f64),
                        )?;
                        // comfy/model_detection.py:915
                        cx.set_item(
                            &cx.get("dit_config")?,
                            &text("rope_t_extrapolation_ratio"),
                            float(0.8333333333333334_f64),
                        )?;
                    }
                }
            }
        }
        // comfy/model_detection.py:917
        cx.set_item(
            &cx.get("dit_config")?,
            &text("extra_h_extrapolation_ratio"),
            float(1.0_f64),
        )?;
        // comfy/model_detection.py:918
        cx.set_item(
            &cx.get("dit_config")?,
            &text("extra_w_extrapolation_ratio"),
            float(1.0_f64),
        )?;
        // comfy/model_detection.py:919
        cx.set_item(
            &cx.get("dit_config")?,
            &text("extra_t_extrapolation_ratio"),
            float(1.0_f64),
        )?;
        // comfy/model_detection.py:920
        cx.set_item(
            &cx.get("dit_config")?,
            &text("rope_enable_fps_modulation"),
            boolean(false),
        )?;
        // comfy/model_detection.py:922
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:924
    if cx.truth(&{
        let detector_tmp_340 = cx.method(
            &text("{}double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_341 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_340, &detector_tmp_341)?
    })? {
        // comfy/model_detection.py:925
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:926
        cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("boogu"))?;
        // comfy/model_detection.py:927
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}x_embedder.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:928
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}single_stream_layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:929
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_double_stream_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}double_stream_layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:930
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_refiner_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}noise_refiner."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:931
        cx.set_item(
            &cx.get("dit_config")?,
            &text("instruction_feat_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}time_caption_embed.caption_embedder.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:932
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:934
    if cx.truth(&{
        let detector_tmp_342 = cx.method(
            &text("{}time_caption_embed.timestep_embedder.linear_1.bias"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_343 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_342, &detector_tmp_343)?
    })? {
        // comfy/model_detection.py:935
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:936
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("omnigen2"),
        )?;
        // comfy/model_detection.py:937
        cx.set_item(
            &cx.get("dit_config")?,
            &text("axes_dim_rope"),
            list(vec![
                integer(40_i128)?,
                integer(40_i128)?,
                integer(40_i128)?,
            ]),
        )?;
        // comfy/model_detection.py:938
        cx.set_item(
            &cx.get("dit_config")?,
            &text("axes_lens"),
            list(vec![
                integer(1024_i128)?,
                integer(1664_i128)?,
                integer(1664_i128)?,
            ]),
        )?;
        // comfy/model_detection.py:939
        cx.set_item(&cx.get("dit_config")?, &text("ffn_dim_multiplier"), none())?;
        // comfy/model_detection.py:940
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            integer(2520_i128)?,
        )?;
        // comfy/model_detection.py:941
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(16_i128)?,
        )?;
        // comfy/model_detection.py:942
        cx.set_item(
            &cx.get("dit_config")?,
            &text("multiple_of"),
            integer(256_i128)?,
        )?;
        // comfy/model_detection.py:943
        cx.set_item(&cx.get("dit_config")?, &text("norm_eps"), float(1e-05_f64))?;
        // comfy/model_detection.py:944
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_attention_heads"),
            integer(21_i128)?,
        )?;
        // comfy/model_detection.py:945
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_kv_heads"),
            integer(7_i128)?,
        )?;
        // comfy/model_detection.py:946
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            integer(32_i128)?,
        )?;
        // comfy/model_detection.py:947
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_refiner_layers"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:948
        cx.set_item(&cx.get("dit_config")?, &text("out_channels"), none())?;
        // comfy/model_detection.py:949
        cx.set_item(
            &cx.get("dit_config")?,
            &text("patch_size"),
            integer(2_i128)?,
        )?;
        // comfy/model_detection.py:950
        cx.set_item(
            &cx.get("dit_config")?,
            &text("text_feat_dim"),
            integer(2048_i128)?,
        )?;
        // comfy/model_detection.py:951
        cx.set_item(
            &cx.get("dit_config")?,
            &text("timestep_scale"),
            float(1000.0_f64),
        )?;
        // comfy/model_detection.py:952
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:954
    if cx.truth(&{
        let detector_tmp_346 = {
            let detector_tmp_347 = cx.method(
                &text("{}transformer_blocks.0.attn.norm_added_q.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_348 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_347, &detector_tmp_348)?
        };
        if cx.truth(&detector_tmp_346)? {
            {
                let detector_tmp_344 = cx.method(
                    &text("{}transformer_blocks.0.img_mlp.w1.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_345 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_344, &detector_tmp_345)?
            }
        } else {
            detector_tmp_346
        }
    })? {
        // comfy/model_detection.py:956
        cx.set(
            "img_in_w",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}img_in.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:957
        cx.set(
            "proj_out_w",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}proj_out.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:958
        cx.set("multi_layer", {
            let detector_tmp_349 = cx.method(
                &text("{}txt_norm.0.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_350 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_349, &detector_tmp_350)?
        });
        // comfy/model_detection.py:959
        if cx.truth(&cx.get("multi_layer")?)? {
            // comfy/model_detection.py:960
            cx.set(
                "enc_hidden_dim",
                cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}txt_norm.0.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
            );
            // comfy/model_detection.py:962
            cx.set(
                "selected_layer_index",
                cx.function(
                    "tuple",
                    vec![cx.function(
                        "range",
                        vec![cx.function(
                            "count_blocks",
                            vec![
                                cx.get("state_dict_keys")?,
                                cx.binary(
                                    "Add",
                                    &cx.method(
                                        &text("{}txt_norm."),
                                        "format",
                                        vec![cx.get("key_prefix")?],
                                        vec![],
                                    )?,
                                    &text("{}."),
                                )?,
                            ],
                            vec![],
                        )?],
                        vec![],
                    )?],
                    vec![],
                )?,
            );
        } else {
            // comfy/model_detection.py:964
            cx.set(
                "enc_hidden_dim",
                cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}txt_norm.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
            );
            // comfy/model_detection.py:965
            cx.set("selected_layer_index", tuple(vec![integer(0_i128)?]));
        }
        // comfy/model_detection.py:967
        return cx.mapping(vec![
            (text("image_model"), text("lens")),
            (
                text("in_channels"),
                cx.index(
                    &cx.attribute(&cx.get("img_in_w")?, "shape")?,
                    &integer(1_i128)?,
                )?,
            ),
            (
                text("out_channels"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(
                        &cx.attribute(&cx.get("proj_out_w")?, "shape")?,
                        &integer(0_i128)?,
                    )?,
                    &integer(4_i128)?,
                )?,
            ),
            (
                text("num_layers"),
                cx.function(
                    "count_blocks",
                    vec![
                        cx.get("state_dict_keys")?,
                        cx.binary(
                            "Add",
                            &cx.method(
                                &text("{}transformer_blocks."),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                            &text("{}."),
                        )?,
                    ],
                    vec![],
                )?,
            ),
            (
                text("num_attention_heads"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(
                        &cx.attribute(&cx.get("img_in_w")?, "shape")?,
                        &integer(0_i128)?,
                    )?,
                    &integer(64_i128)?,
                )?,
            ),
            (text("enc_hidden_dim"), cx.get("enc_hidden_dim")?),
            (text("multi_layer_encoder_feature"), cx.get("multi_layer")?),
            (
                text("selected_layer_index"),
                cx.get("selected_layer_index")?,
            ),
        ]);
    }
    // comfy/model_detection.py:978
    if cx.truth(&{
        let detector_tmp_359 = {
            let detector_tmp_360 = cx.method(
                &text("{}txt_norm.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_361 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_360, &detector_tmp_361)?
        };
        if cx.truth(&detector_tmp_359)? {
            {
                let detector_tmp_356 = {
                    let detector_tmp_357 = cx.method(
                        &text("{}proj_out.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_358 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_357, &detector_tmp_358)?
                };
                if cx.truth(&detector_tmp_356)? {
                    {
                        let detector_tmp_353 = {
                            let detector_tmp_354 = cx.index(
                                &cx.attribute(
                                    &cx.index(
                                        &cx.get("state_dict")?,
                                        &cx.method(
                                            &text("{}txt_norm.weight"),
                                            "format",
                                            vec![cx.get("key_prefix")?],
                                            vec![],
                                        )?,
                                    )?,
                                    "shape",
                                )?,
                                &integer(0_i128)?,
                            )?;
                            let detector_tmp_355 = integer(2560_i128)?;
                            cx.compare("Eq", &detector_tmp_354, &detector_tmp_355)?
                        };
                        if cx.truth(&detector_tmp_353)? {
                            {
                                let detector_tmp_351 = cx.index(
                                    &cx.attribute(
                                        &cx.index(
                                            &cx.get("state_dict")?,
                                            &cx.method(
                                                &text("{}proj_out.weight"),
                                                "format",
                                                vec![cx.get("key_prefix")?],
                                                vec![],
                                            )?,
                                        )?,
                                        "shape",
                                    )?,
                                    &integer(0_i128)?,
                                )?;
                                let detector_tmp_352 = integer(128_i128)?;
                                cx.compare("Eq", &detector_tmp_351, &detector_tmp_352)?
                            }
                        } else {
                            detector_tmp_353
                        }
                    }
                } else {
                    detector_tmp_356
                }
            }
        } else {
            detector_tmp_359
        }
    })? {
        // comfy/model_detection.py:979
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:980
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("mage_flow"),
        )?;
        // comfy/model_detection.py:981
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            integer(128_i128)?,
        )?;
        // comfy/model_detection.py:982
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}transformer_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:983
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:985
    cx.set(
        "qwen_image21_keys",
        list(vec![
            text("txt_in.text_norm.weight"),
            text("modulation.1.weight"),
            text("transformer_blocks.0.attn.norm_q.weight"),
            text("img_in.weight"),
            text("proj_out.weight"),
        ]),
    );
    // comfy/model_detection.py:986
    if cx.truth(&{
        let detector_tmp_364 = cx.comprehension(
            cx.iter(&cx.get("qwen_image21_keys")?)?,
            "all",
            |cx, item| {
                cx.set("k", item);
                Ok(Some({
                    let detector_tmp_365 = cx.method(
                        &text("{}{}"),
                        "format",
                        vec![cx.get("key_prefix")?, cx.get("k")?],
                        vec![],
                    )?;
                    let detector_tmp_366 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_365, &detector_tmp_366)?
                }))
            },
        )?;
        if cx.truth(&detector_tmp_364)? {
            cx.comprehension(
                cx.iter(&tuple(vec![text("gate_up"), text("proj")]))?,
                "any",
                |cx, item| {
                    cx.set("k", item);
                    Ok(Some({
                        let detector_tmp_362 = cx.method(
                            &text("{}transformer_blocks.0.img_mlp.{}.weight"),
                            "format",
                            vec![cx.get("key_prefix")?, cx.get("k")?],
                            vec![],
                        )?;
                        let detector_tmp_363 = cx.get("state_dict_keys")?;
                        cx.compare("In", &detector_tmp_362, &detector_tmp_363)?
                    }))
                },
            )?
        } else {
            detector_tmp_364
        }
    })? {
        // comfy/model_detection.py:987
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:988
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("qwen_image21"),
        )?;
        // comfy/model_detection.py:989
        cx.set(
            "head_dim",
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}transformer_blocks.0.attn.norm_q.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        );
        // comfy/model_detection.py:990
        cx.set(
            "inner_dim",
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}img_in.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        );
        // comfy/model_detection.py:991
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}img_in.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:992
        cx.set_item(
            &cx.get("dit_config")?,
            &text("out_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}proj_out.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:993
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}transformer_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:994
        cx.set_item(
            &cx.get("dit_config")?,
            &text("attention_head_dim"),
            cx.get("head_dim")?,
        )?;
        // comfy/model_detection.py:995
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_attention_heads"),
            cx.binary("FloorDiv", &cx.get("inner_dim")?, &cx.get("head_dim")?)?,
        )?;
        // comfy/model_detection.py:996
        cx.set_item(
            &cx.get("dit_config")?,
            &text("context_in_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}txt_in.text_norm.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:998
        cx.set(
            "gate_up",
            cx.method(
                &cx.get("state_dict")?,
                "get",
                vec![
                    cx.method(
                        &text("{}transformer_blocks.0.img_mlp.gate_up.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                    none(),
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:999
        if cx.truth(&{
            let detector_tmp_367 = cx.get("gate_up")?;
            let detector_tmp_368 = none();
            cx.compare("IsNot", &detector_tmp_367, &detector_tmp_368)?
        })? {
            // comfy/model_detection.py:1000
            cx.set_item(
                &cx.get("dit_config")?,
                &text("mlp_ratio"),
                cx.binary(
                    "FloorDiv",
                    &cx.binary(
                        "FloorDiv",
                        &cx.index(
                            &cx.attribute(&cx.get("gate_up")?, "shape")?,
                            &integer(0_i128)?,
                        )?,
                        &integer(2_i128)?,
                    )?,
                    &cx.get("inner_dim")?,
                )?,
            )?;
        } else {
            // comfy/model_detection.py:1002
            cx.set_item(
                &cx.get("dit_config")?,
                &text("mlp_ratio"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(
                        &cx.attribute(
                            &cx.index(
                                &cx.get("state_dict")?,
                                &cx.method(
                                    &text("{}transformer_blocks.0.img_mlp.proj.weight"),
                                    "format",
                                    vec![cx.get("key_prefix")?],
                                    vec![],
                                )?,
                            )?,
                            "shape",
                        )?,
                        &integer(0_i128)?,
                    )?,
                    &cx.get("inner_dim")?,
                )?,
            )?;
        }
        // comfy/model_detection.py:1003
        cx.set_item(&cx.get("dit_config")?, &text("fused_mlp"), {
            let detector_tmp_369 = cx.get("gate_up")?;
            let detector_tmp_370 = none();
            cx.compare("IsNot", &detector_tmp_369, &detector_tmp_370)?
        })?;
        // comfy/model_detection.py:1004
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1006
    if cx.truth(&{
        let detector_tmp_371 = cx.method(
            &text("{}txt_norm.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_372 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_371, &detector_tmp_372)?
    })? {
        // comfy/model_detection.py:1007
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1008
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("qwen_image"),
        )?;
        // comfy/model_detection.py:1009
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}img_in.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1010
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}transformer_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:1011
        if cx.truth(&{
            let detector_tmp_373 = cx.method(
                &text("{}__index_timestep_zero__"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_374 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_373, &detector_tmp_374)?
        })? {
            // comfy/model_detection.py:1012
            cx.set_item(
                &cx.get("dit_config")?,
                &text("default_ref_method"),
                text("index_timestep_zero"),
            )?;
        }
        // comfy/model_detection.py:1013
        if cx.truth(&{
            let detector_tmp_375 = cx.method(
                &text("{}time_text_embed.addition_t_embedding.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_376 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_375, &detector_tmp_376)?
        })? {
            // comfy/model_detection.py:1014
            cx.set_item(
                &cx.get("dit_config")?,
                &text("use_additional_t_cond"),
                boolean(true),
            )?;
            // comfy/model_detection.py:1015
            cx.set_item(
                &cx.get("dit_config")?,
                &text("default_ref_method"),
                text("negative_index"),
            )?;
        }
        // comfy/model_detection.py:1016
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1018
    if cx.truth(&{
        let detector_tmp_377 = cx.method(
            &text("{}embed_image_indicator.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_378 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_377, &detector_tmp_378)?
    })? {
        // comfy/model_detection.py:1019
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1020
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("ideogram4"),
        )?;
        // comfy/model_detection.py:1021
        cx.set_item(
            &cx.get("dit_config")?,
            &text("in_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}input_proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1022
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:1023
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1025
    if cx.truth(&{
        let detector_tmp_381 = {
            let detector_tmp_382 = cx.method(
                &text("{}shared_block_modulation.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_383 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_382, &detector_tmp_383)?
        };
        if cx.truth(&detector_tmp_381)? {
            {
                let detector_tmp_379 = cx.method(
                    &text("{}layers.0.proj.ffn_up_gate.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_380 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_379, &detector_tmp_380)?
            }
        } else {
            detector_tmp_381
        }
    })? {
        // comfy/model_detection.py:1026
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1027
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("moss_image"),
        )?;
        // comfy/model_detection.py:1028
        cx.set_item(
            &cx.get("dit_config")?,
            &text("dim"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}shared_block_modulation.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &integer(6_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1029
        cx.set_item(
            &cx.get("dit_config")?,
            &text("ffn_dim"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}layers.0.proj.ffn_up_gate.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &integer(2_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1030
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(&cx.get("dit_config")?, &text("dim"))?,
                &integer(128_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1031
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:1032
        cx.set_item(&cx.get("dit_config")?, &text("fused_qkv"), {
            let detector_tmp_384 = cx.method(
                &text("{}layers.0.proj.qkv.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_385 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_384, &detector_tmp_385)?
        })?;
        // comfy/model_detection.py:1033
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1035
    if cx.truth(&{
        let detector_tmp_386 = cx.method(
            &text("{}txtfusion.projector.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_387 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_386, &detector_tmp_387)?
    })? {
        // comfy/model_detection.py:1036
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1037
        cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("krea2"))?;
        // comfy/model_detection.py:1038
        cx.set("head_dim", integer(128_i128)?);
        // comfy/model_detection.py:1039
        cx.set(
            "first_w",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}first.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:1040
        cx.set_item(
            &cx.get("dit_config")?,
            &text("features"),
            cx.index(
                &cx.attribute(&cx.get("first_w")?, "shape")?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1041
        cx.set_item(
            &cx.get("dit_config")?,
            &text("channels"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(&cx.get("first_w")?, "shape")?,
                    &integer(1_i128)?,
                )?,
                &cx.binary("Mult", &integer(2_i128)?, &integer(2_i128)?)?,
            )?,
        )?;
        // comfy/model_detection.py:1042
        cx.set_item(&cx.get("dit_config")?, &text("patch"), integer(2_i128)?)?;
        // comfy/model_detection.py:1043
        cx.set_item(
            &cx.get("dit_config")?,
            &text("layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:1044
        cx.set_item(
            &cx.get("dit_config")?,
            &text("heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}blocks.0.attn.wq.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &cx.get("head_dim")?,
            )?,
        )?;
        // comfy/model_detection.py:1045
        cx.set_item(
            &cx.get("dit_config")?,
            &text("kvheads"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}blocks.0.attn.wk.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &cx.get("head_dim")?,
            )?,
        )?;
        // comfy/model_detection.py:1046
        cx.set_item(
            &cx.get("dit_config")?,
            &text("txtlayers"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}txtfusion.projector.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1047
        cx.set_item(
            &cx.get("dit_config")?,
            &text("txtdim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}txtfusion.layerwise_blocks.0.prenorm.scale"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1048
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1050
    if cx.truth(&{
        let detector_tmp_388 = cx.method(
            &text("{}visual_transformer_blocks.0.cross_attention.key_norm.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_389 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_388, &detector_tmp_389)?
    })? {
        // comfy/model_detection.py:1051
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1052
        cx.set(
            "model_dim",
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}visual_embeddings.in_layer.bias"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        );
        // comfy/model_detection.py:1053
        cx.set_item(
            &cx.get("dit_config")?,
            &text("model_dim"),
            cx.get("model_dim")?,
        )?;
        // comfy/model_detection.py:1054
        if cx.truth(&{
            let detector_tmp_390 = cx.get("model_dim")?;
            let detector_tmp_391 = list(vec![integer(4096_i128)?, integer(2560_i128)?]);
            cx.compare("In", &detector_tmp_390, &detector_tmp_391)?
        })? {
            // comfy/model_detection.py:1055
            cx.set_item(
                &cx.get("dit_config")?,
                &text("axes_dims"),
                tuple(vec![
                    integer(32_i128)?,
                    integer(48_i128)?,
                    integer(48_i128)?,
                ]),
            )?;
            // comfy/model_detection.py:1056
            if cx.truth(&{
                let detector_tmp_392 = cx.get("model_dim")?;
                let detector_tmp_393 = integer(2560_i128)?;
                cx.compare("Eq", &detector_tmp_392, &detector_tmp_393)?
            })? {
                // comfy/model_detection.py:1057
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("rope_scale_factor"),
                    tuple(vec![float(1.0_f64), float(1.0_f64), float(1.0_f64)]),
                )?;
            }
        } else {
            // comfy/model_detection.py:1058
            if cx.truth(&{
                let detector_tmp_394 = cx.get("model_dim")?;
                let detector_tmp_395 = integer(1792_i128)?;
                cx.compare("Eq", &detector_tmp_394, &detector_tmp_395)?
            })? {
                // comfy/model_detection.py:1059
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("axes_dims"),
                    tuple(vec![
                        integer(16_i128)?,
                        integer(24_i128)?,
                        integer(24_i128)?,
                    ]),
                )?;
            }
        }
        // comfy/model_detection.py:1060
        cx.set_item(
            &cx.get("dit_config")?,
            &text("time_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}time_embeddings.in_layer.bias"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1061
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("kandinsky5"),
        )?;
        // comfy/model_detection.py:1062
        cx.set_item(
            &cx.get("dit_config")?,
            &text("ff_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}visual_transformer_blocks.0.feed_forward.in_layer.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1063
        cx.set_item(
            &cx.get("dit_config")?,
            &text("visual_embed_dim"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}visual_embeddings.in_layer.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1064
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_text_blocks"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}text_transformer_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:1065
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_visual_blocks"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}visual_transformer_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:1066
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1068
    if cx.truth(&{
        let detector_tmp_396 = cx.method(
            &text("{}encoder.lyric_encoder.layers.0.input_layernorm.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_397 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_396, &detector_tmp_397)?
    })? {
        // comfy/model_detection.py:1069
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1070
        cx.set_item(&cx.get("dit_config")?, &text("audio_model"), text("ace1.5"))?;
        // comfy/model_detection.py:1071
        cx.set("head_dim", integer(128_i128)?);
        // comfy/model_detection.py:1072
        cx.set_item(
            &cx.get("dit_config")?,
            &text("hidden_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}decoder.layers.0.self_attn_norm.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1073
        cx.set_item(
            &cx.get("dit_config")?,
            &text("intermediate_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}decoder.layers.0.mlp.gate_proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1074
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}decoder.layers.0.self_attn.q_proj.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &cx.get("head_dim")?,
            )?,
        )?;
        // comfy/model_detection.py:1076
        cx.set_item(
            &cx.get("dit_config")?,
            &text("encoder_hidden_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}encoder.lyric_encoder.layers.0.input_layernorm.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1077
        cx.set_item(
            &cx.get("dit_config")?,
            &text("encoder_num_heads"),
            cx.binary(
                "FloorDiv",
                &cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}encoder.lyric_encoder.layers.0.self_attn.q_proj.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
                &cx.get("head_dim")?,
            )?,
        )?;
        // comfy/model_detection.py:1078
        cx.set_item(
            &cx.get("dit_config")?,
            &text("encoder_intermediate_size"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}encoder.lyric_encoder.layers.0.mlp.gate_proj.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1079
        cx.set_item(
            &cx.get("dit_config")?,
            &text("num_dit_layers"),
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}decoder.layers."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        )?;
        // comfy/model_detection.py:1080
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1082
    if cx.truth(&{
        let detector_tmp_398 = cx.method(
            &text("{}encoder.pan_blocks.1.cv4.conv.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_399 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_398, &detector_tmp_399)?
    })? {
        // comfy/model_detection.py:1083
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1084
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("RT_DETR_v4"),
        )?;
        // comfy/model_detection.py:1085
        cx.set_item(
            &cx.get("dit_config")?,
            &text("enc_h"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}encoder.pan_blocks.1.cv4.conv.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1086
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1089
    if cx.truth(&{
        let detector_tmp_400 = cx.method(
            &text("{}backbone.embeddings.patch_embeddings.projection.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_401 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_400, &detector_tmp_401)?
    })? {
        // comfy/model_detection.py:1090
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1091
        cx.set_item(
            &cx.get("dit_config")?,
            &text("image_model"),
            text("DepthAnything3"),
        )?;
        // comfy/model_detection.py:1093
        cx.set(
            "patch_w",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}backbone.embeddings.patch_embeddings.projection.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:1094
        cx.set(
            "embed_dim",
            cx.index(
                &cx.attribute(&cx.get("patch_w")?, "shape")?,
                &integer(0_i128)?,
            )?,
        );
        // comfy/model_detection.py:1095
        cx.set(
            "depth",
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}backbone.encoder.layer."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}."),
                    )?,
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1098
        cx.set(
            "backbone_name",
            cx.method(
                &cx.mapping(vec![
                    (integer(384_i128)?, text("vits")),
                    (integer(768_i128)?, text("vitb")),
                    (integer(1024_i128)?, text("vitl")),
                    (integer(1536_i128)?, text("vitg")),
                ])?,
                "get",
                vec![cx.get("embed_dim")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1099
        if cx.truth(&{
            let detector_tmp_402 = cx.get("backbone_name")?;
            let detector_tmp_403 = none();
            cx.compare("Is", &detector_tmp_402, &detector_tmp_403)?
        })? {
            // comfy/model_detection.py:1100
            return Ok(none());
        }
        // comfy/model_detection.py:1101
        cx.set_item(
            &cx.get("dit_config")?,
            &text("backbone_name"),
            cx.get("backbone_name")?,
        )?;
        // comfy/model_detection.py:1104
        cx.set("has_camera_token", {
            let detector_tmp_404 = cx.method(
                &text("{}backbone.embeddings.camera_token"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_405 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_404, &detector_tmp_405)?
        });
        // comfy/model_detection.py:1106
        cx.set(
            "qknorm_indices",
            cx.comprehension(
                cx.iter(&cx.function("range", vec![cx.get("depth")?], vec![])?)?,
                "list",
                |cx, item| {
                    cx.set("i", item);
                    if !cx.truth(&{
                        let detector_tmp_406 = cx.method(
                            &text("{}backbone.encoder.layer.{}.attention.q_norm.weight"),
                            "format",
                            vec![cx.get("key_prefix")?, cx.get("i")?],
                            vec![],
                        )?;
                        let detector_tmp_407 = cx.get("state_dict_keys")?;
                        cx.compare("In", &detector_tmp_406, &detector_tmp_407)?
                    })? {
                        return Ok(None);
                    }
                    Ok(Some(cx.get("i")?))
                },
            )?,
        );
        // comfy/model_detection.py:1110
        cx.set(
            "qknorm_start",
            if cx.truth(&cx.get("qknorm_indices")?)? {
                cx.index(&cx.get("qknorm_indices")?, &integer(0_i128)?)?
            } else {
                cx.unary("USub", &integer(1_i128)?)?
            },
        );
        // comfy/model_detection.py:1114
        if cx.truth(&cx.get("has_camera_token")?)? {
            // comfy/model_detection.py:1115
            cx.set_item(
                &cx.get("dit_config")?,
                &text("alt_start"),
                cx.get("qknorm_start")?,
            )?;
            // comfy/model_detection.py:1116
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_start"),
                cx.get("qknorm_start")?,
            )?;
            // comfy/model_detection.py:1117
            cx.set_item(
                &cx.get("dit_config")?,
                &text("qknorm_start"),
                cx.get("qknorm_start")?,
            )?;
            // comfy/model_detection.py:1118
            cx.set_item(&cx.get("dit_config")?, &text("cat_token"), boolean(true))?;
        } else {
            // comfy/model_detection.py:1120
            cx.set_item(
                &cx.get("dit_config")?,
                &text("alt_start"),
                cx.unary("USub", &integer(1_i128)?)?,
            )?;
            // comfy/model_detection.py:1121
            cx.set_item(
                &cx.get("dit_config")?,
                &text("rope_start"),
                cx.unary("USub", &integer(1_i128)?)?,
            )?;
            // comfy/model_detection.py:1122
            cx.set_item(
                &cx.get("dit_config")?,
                &text("qknorm_start"),
                cx.unary("USub", &integer(1_i128)?)?,
            )?;
            // comfy/model_detection.py:1123
            cx.set_item(&cx.get("dit_config")?, &text("cat_token"), boolean(false))?;
        }
        // comfy/model_detection.py:1126
        cx.set("has_aux", {
            let detector_tmp_408 = cx.method(
                &text("{}head.scratch.refinenet1_aux.out_conv.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_409 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_408, &detector_tmp_409)?
        });
        // comfy/model_detection.py:1127
        cx.set_item(
            &cx.get("dit_config")?,
            &text("head_dim_in"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}head.projects.0.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1128
        cx.set_item(
            &cx.get("dit_config")?,
            &text("head_features"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}head.scratch.refinenet1.out_conv.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        )?;
        // comfy/model_detection.py:1129
        cx.set_item(
            &cx.get("dit_config")?,
            &text("head_out_channels"),
            cx.comprehension(
                cx.iter(&cx.function("range", vec![integer(4_i128)?], vec![])?)?,
                "list",
                |cx, item| {
                    cx.set("i", item);
                    Ok(Some(cx.index(
                        &cx.attribute(
                            &cx.index(
                                &cx.get("state_dict")?,
                                &cx.method(
                                    &text("{}head.projects.{}.weight"),
                                    "format",
                                    vec![cx.get("key_prefix")?, cx.get("i")?],
                                    vec![],
                                )?,
                            )?,
                            "shape",
                        )?,
                        &integer(0_i128)?,
                    )?))
                },
            )?,
        )?;
        // comfy/model_detection.py:1133
        if cx.truth(&cx.get("has_aux")?)? {
            // comfy/model_detection.py:1135
            cx.set_item(&cx.get("dit_config")?, &text("head_type"), text("dualdpt"))?;
            // comfy/model_detection.py:1136
            cx.set_item(
                &cx.get("dit_config")?,
                &text("head_output_dim"),
                integer(2_i128)?,
            )?;
            // comfy/model_detection.py:1137
            cx.set_item(
                &cx.get("dit_config")?,
                &text("head_use_sky_head"),
                boolean(false),
            )?;
        } else {
            // comfy/model_detection.py:1139
            cx.set_item(&cx.get("dit_config")?, &text("head_type"), text("dpt"))?;
            // comfy/model_detection.py:1140
            cx.set_item(
                &cx.get("dit_config")?,
                &text("head_output_dim"),
                cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &cx.method(
                                &text("{}head.scratch.output_conv2.2.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                        )?,
                        "shape",
                    )?,
                    &integer(0_i128)?,
                )?,
            )?;
            // comfy/model_detection.py:1143
            cx.set_item(&cx.get("dit_config")?, &text("head_use_sky_head"), {
                let detector_tmp_410 = cx.method(
                    &text("{}head.scratch.sky_output_conv2.0.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?;
                let detector_tmp_411 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_410, &detector_tmp_411)?
            })?;
        }
        // comfy/model_detection.py:1148
        if cx.truth(&{
            let detector_tmp_412 = cx.get("depth")?;
            let detector_tmp_413 = integer(24_i128)?;
            cx.compare("GtE", &detector_tmp_412, &detector_tmp_413)?
        })? {
            // comfy/model_detection.py:1150
            if cx.truth(&cx.get("has_aux")?)? {
                // comfy/model_detection.py:1151
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("out_layers"),
                    list(vec![
                        integer(11_i128)?,
                        integer(15_i128)?,
                        integer(19_i128)?,
                        integer(23_i128)?,
                    ]),
                )?;
            } else {
                // comfy/model_detection.py:1153
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("out_layers"),
                    list(vec![
                        integer(4_i128)?,
                        integer(11_i128)?,
                        integer(17_i128)?,
                        integer(23_i128)?,
                    ]),
                )?;
            }
        } else {
            // comfy/model_detection.py:1156
            cx.set_item(
                &cx.get("dit_config")?,
                &text("out_layers"),
                list(vec![
                    integer(5_i128)?,
                    integer(7_i128)?,
                    integer(9_i128)?,
                    integer(11_i128)?,
                ]),
            )?;
        }
        // comfy/model_detection.py:1159
        cx.set("has_cam_enc", {
            let detector_tmp_414 = cx.method(
                &text("{}cam_enc.token_norm.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_415 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_414, &detector_tmp_415)?
        });
        // comfy/model_detection.py:1160
        cx.set("has_cam_dec", {
            let detector_tmp_416 = cx.method(
                &text("{}cam_dec.fc_t.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_417 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_416, &detector_tmp_417)?
        });
        // comfy/model_detection.py:1161
        cx.set_item(
            &cx.get("dit_config")?,
            &text("has_cam_enc"),
            cx.get("has_cam_enc")?,
        )?;
        // comfy/model_detection.py:1162
        cx.set_item(
            &cx.get("dit_config")?,
            &text("has_cam_dec"),
            cx.get("has_cam_dec")?,
        )?;
        // comfy/model_detection.py:1163
        if cx.truth(&cx.get("has_cam_enc")?)? {
            // comfy/model_detection.py:1164
            cx.set(
                "cam_enc_w",
                cx.method(
                    &cx.get("state_dict")?,
                    "get",
                    vec![cx.method(
                        &text("{}cam_enc.pose_branch.fc2.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:1167
            if cx.truth(&{
                let detector_tmp_418 = cx.get("cam_enc_w")?;
                let detector_tmp_419 = none();
                cx.compare("IsNot", &detector_tmp_418, &detector_tmp_419)?
            })? {
                // comfy/model_detection.py:1168
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("cam_dim_out"),
                    cx.index(
                        &cx.attribute(&cx.get("cam_enc_w")?, "shape")?,
                        &integer(0_i128)?,
                    )?,
                )?;
            }
        }
        // comfy/model_detection.py:1169
        if cx.truth(&cx.get("has_cam_dec")?)? {
            // comfy/model_detection.py:1170
            cx.set(
                "cam_dec_w",
                cx.method(
                    &cx.get("state_dict")?,
                    "get",
                    vec![cx.method(
                        &text("{}cam_dec.fc_t.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:1173
            if cx.truth(&{
                let detector_tmp_420 = cx.get("cam_dec_w")?;
                let detector_tmp_421 = none();
                cx.compare("IsNot", &detector_tmp_420, &detector_tmp_421)?
            })? {
                // comfy/model_detection.py:1174
                cx.set_item(
                    &cx.get("dit_config")?,
                    &text("cam_dec_dim_in"),
                    cx.index(
                        &cx.attribute(&cx.get("cam_dec_w")?, "shape")?,
                        &integer(1_i128)?,
                    )?,
                )?;
            }
        }
        // comfy/model_detection.py:1175
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1177
    if cx.truth(&{
        let detector_tmp_422 = cx.method(
            &text("{}layers.0.mlp.linear_fc2.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_423 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_422, &detector_tmp_423)?
    })? {
        // comfy/model_detection.py:1178
        cx.set("dit_config", cx.mapping(vec![])?);
        // comfy/model_detection.py:1179
        cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("ernie"))?;
        // comfy/model_detection.py:1180
        return cx.get("dit_config");
    }
    // comfy/model_detection.py:1182
    if cx.truth(&{
        let detector_tmp_424 =
            text("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight");
        let detector_tmp_425 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_424, &detector_tmp_425)?
    })? {
        // comfy/model_detection.py:1183
        if cx.truth(&{
            let detector_tmp_426 = text("detector.transformer.decoder.query_embed.weight");
            let detector_tmp_427 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_426, &detector_tmp_427)?
        })? {
            // comfy/model_detection.py:1184
            cx.set("dit_config", cx.mapping(vec![])?);
            // comfy/model_detection.py:1185
            cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("SAM3"))?;
            // comfy/model_detection.py:1186
            if cx.truth(&{
                let detector_tmp_428 =
                    text("detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight");
                let detector_tmp_429 = cx.get("state_dict_keys")?;
                cx.compare("In", &detector_tmp_428, &detector_tmp_429)?
            })? {
                // comfy/model_detection.py:1187
                cx.set_item(&cx.get("dit_config")?, &text("image_model"), text("SAM31"))?;
            }
            // comfy/model_detection.py:1188
            return cx.get("dit_config");
        }
    }
    // comfy/model_detection.py:1190
    if cx.truth(&{
        let detector_tmp_441 = {
            let detector_tmp_442 = cx.method(
                &text("{}double_blocks.0.attn.img_attn_qkv.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_443 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_442, &detector_tmp_443)?
        };
        if cx.truth(&detector_tmp_441)? {
            {
                let detector_tmp_438 = {
                    let detector_tmp_439 = cx.method(
                        &text("{}double_blocks.0.attn.img_attn_q_norm.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?;
                    let detector_tmp_440 = cx.get("state_dict_keys")?;
                    cx.compare("In", &detector_tmp_439, &detector_tmp_440)?
                };
                if cx.truth(&detector_tmp_438)? {
                    {
                        let detector_tmp_435 = {
                            let detector_tmp_436 = cx.method(
                                &text("{}condition_embedder.time_embedder.linear_1.weight"),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?;
                            let detector_tmp_437 = cx.get("state_dict_keys")?;
                            cx.compare("In", &detector_tmp_436, &detector_tmp_437)?
                        };
                        if cx.truth(&detector_tmp_435)? {
                            {
                                let detector_tmp_432 = {
                                    let detector_tmp_433 = cx.method(
                                        &text("{}img_in.weight"),
                                        "format",
                                        vec![cx.get("key_prefix")?],
                                        vec![],
                                    )?;
                                    let detector_tmp_434 = cx.get("state_dict_keys")?;
                                    cx.compare("In", &detector_tmp_433, &detector_tmp_434)?
                                };
                                if cx.truth(&detector_tmp_432)? {
                                    {
                                        let detector_tmp_430 = cx.function(
                                            "len",
                                            vec![cx.attribute(
                                                &cx.index(
                                                    &cx.get("state_dict")?,
                                                    &cx.method(
                                                        &text("{}img_in.weight"),
                                                        "format",
                                                        vec![cx.get("key_prefix")?],
                                                        vec![],
                                                    )?,
                                                )?,
                                                "shape",
                                            )?],
                                            vec![],
                                        )?;
                                        let detector_tmp_431 = integer(5_i128)?;
                                        cx.compare("Eq", &detector_tmp_430, &detector_tmp_431)?
                                    }
                                } else {
                                    detector_tmp_432
                                }
                            }
                        } else {
                            detector_tmp_435
                        }
                    }
                } else {
                    detector_tmp_438
                }
            }
        } else {
            detector_tmp_441
        }
    })? {
        // comfy/model_detection.py:1197
        cx.set(
            "img_in",
            cx.index(
                &cx.get("state_dict")?,
                &cx.method(
                    &text("{}img_in.weight"),
                    "format",
                    vec![cx.get("key_prefix")?],
                    vec![],
                )?,
            )?,
        );
        // comfy/model_detection.py:1198
        cx.set(
            "head_dim",
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &cx.method(
                            &text("{}double_blocks.0.attn.img_attn_q_norm.weight"),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                    )?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        );
        // comfy/model_detection.py:1199
        return cx.mapping(vec![
            (text("image_model"), text("joyimage")),
            (
                text("in_channels"),
                cx.index(
                    &cx.attribute(&cx.get("img_in")?, "shape")?,
                    &integer(1_i128)?,
                )?,
            ),
            (
                text("hidden_size"),
                cx.index(
                    &cx.attribute(&cx.get("img_in")?, "shape")?,
                    &integer(0_i128)?,
                )?,
            ),
            (
                text("patch_size"),
                cx.function(
                    "list",
                    vec![cx.slice(
                        &cx.attribute(&cx.get("img_in")?, "shape")?,
                        Some(integer(2_i128)?),
                        None,
                        None,
                    )?],
                    vec![],
                )?,
            ),
            (
                text("num_layers"),
                cx.function(
                    "count_blocks",
                    vec![
                        cx.get("state_dict_keys")?,
                        cx.binary(
                            "Add",
                            &cx.method(
                                &text("{}double_blocks."),
                                "format",
                                vec![cx.get("key_prefix")?],
                                vec![],
                            )?,
                            &text("{}."),
                        )?,
                    ],
                    vec![],
                )?,
            ),
            (
                text("num_attention_heads"),
                cx.binary(
                    "FloorDiv",
                    &cx.index(
                        &cx.attribute(&cx.get("img_in")?, "shape")?,
                        &integer(0_i128)?,
                    )?,
                    &cx.get("head_dim")?,
                )?,
            ),
            (text("text_dim"), integer(4096_i128)?),
        ]);
    }
    // comfy/model_detection.py:1209
    if cx.truth(&cx.comprehension(
        cx.iter(&tuple(vec![
            text("vae2llm.weight"),
            text("llm2vae.weight"),
            text("latent_pos_embed.pe"),
            text("model.layers.0.self_attn.qkv_proj.weight"),
            text("time_embedder.mlp.0.weight"),
        ]))?,
        "all",
        |cx, item| {
            cx.set("key", item);
            Ok(Some({
                let detector_tmp_444 = cx.binary("Add", &cx.get("key_prefix")?, &cx.get("key")?)?;
                let detector_tmp_445 = cx.get("state_dict")?;
                cx.compare("In", &detector_tmp_444, &detector_tmp_445)?
            }))
        },
    )?)? {
        // comfy/model_detection.py:1213
        return cx.mapping(vec![(text("audio_model"), text("yue2"))]);
    }
    // comfy/model_detection.py:1215
    if cx.truth(&{
        let detector_tmp_446 = cx.method(
            &text("{}input_blocks.0.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_447 = cx.get("state_dict_keys")?;
        cx.compare("NotIn", &detector_tmp_446, &detector_tmp_447)?
    })? {
        // comfy/model_detection.py:1216
        return Ok(none());
    }
    // comfy/model_detection.py:1218
    cx.set(
        "unet_config",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1225
    cx.set(
        "y_input",
        cx.method(
            &text("{}label_emb.0.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:1226
    if cx.truth(&{
        let detector_tmp_448 = cx.get("y_input")?;
        let detector_tmp_449 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_448, &detector_tmp_449)?
    })? {
        // comfy/model_detection.py:1227
        cx.set_item(
            &cx.get("unet_config")?,
            &text("num_classes"),
            text("sequential"),
        )?;
        // comfy/model_detection.py:1228
        cx.set_item(
            &cx.get("unet_config")?,
            &text("adm_in_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(&cx.get("state_dict")?, &cx.get("y_input")?)?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
    } else {
        // comfy/model_detection.py:1230
        cx.set_item(&cx.get("unet_config")?, &text("adm_in_channels"), none())?;
    }
    // comfy/model_detection.py:1232
    cx.set(
        "model_channels",
        cx.index(
            &cx.attribute(
                &cx.index(
                    &cx.get("state_dict")?,
                    &cx.method(
                        &text("{}input_blocks.0.0.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                )?,
                "shape",
            )?,
            &integer(0_i128)?,
        )?,
    );
    // comfy/model_detection.py:1233
    cx.set(
        "in_channels",
        cx.index(
            &cx.attribute(
                &cx.index(
                    &cx.get("state_dict")?,
                    &cx.method(
                        &text("{}input_blocks.0.0.weight"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                )?,
                "shape",
            )?,
            &integer(1_i128)?,
        )?,
    );
    // comfy/model_detection.py:1235
    cx.set(
        "out_key",
        cx.method(
            &text("{}out.2.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:1236
    if cx.truth(&{
        let detector_tmp_450 = cx.get("out_key")?;
        let detector_tmp_451 = cx.get("state_dict")?;
        cx.compare("In", &detector_tmp_450, &detector_tmp_451)?
    })? {
        // comfy/model_detection.py:1237
        cx.set(
            "out_channels",
            cx.index(
                &cx.attribute(
                    &cx.index(&cx.get("state_dict")?, &cx.get("out_key")?)?,
                    "shape",
                )?,
                &integer(0_i128)?,
            )?,
        );
    } else {
        // comfy/model_detection.py:1239
        cx.set("out_channels", integer(4_i128)?);
    }
    // comfy/model_detection.py:1241
    cx.set("num_res_blocks", list(vec![]));
    // comfy/model_detection.py:1242
    cx.set("channel_mult", list(vec![]));
    // comfy/model_detection.py:1243
    cx.set("transformer_depth", list(vec![]));
    // comfy/model_detection.py:1244
    cx.set("transformer_depth_output", list(vec![]));
    // comfy/model_detection.py:1245
    cx.set("context_dim", none());
    // comfy/model_detection.py:1246
    cx.set("use_linear_in_transformer", boolean(false));
    // comfy/model_detection.py:1248
    cx.set("video_model", boolean(false));
    // comfy/model_detection.py:1249
    cx.set("video_model_cross", boolean(false));
    // comfy/model_detection.py:1251
    cx.set("current_res", integer(1_i128)?);
    // comfy/model_detection.py:1252
    cx.set("count", integer(0_i128)?);
    // comfy/model_detection.py:1254
    cx.set("last_res_blocks", integer(0_i128)?);
    // comfy/model_detection.py:1255
    cx.set("last_channel_mult", integer(0_i128)?);
    // comfy/model_detection.py:1257
    cx.set(
        "input_block_count",
        cx.function(
            "count_blocks",
            vec![
                cx.get("state_dict_keys")?,
                cx.binary(
                    "Add",
                    &cx.method(
                        &text("{}input_blocks"),
                        "format",
                        vec![cx.get("key_prefix")?],
                        vec![],
                    )?,
                    &text(".{}."),
                )?,
            ],
            vec![],
        )?,
    );
    // comfy/model_detection.py:1258
    for detector_item in
        cx.iter(&cx.function("range", vec![cx.get("input_block_count")?], vec![])?)?
    {
        cx.set("count", detector_item); // comfy/model_detection.py:1259
        cx.set(
            "prefix",
            cx.method(
                &text("{}input_blocks.{}."),
                "format",
                vec![cx.get("key_prefix")?, cx.get("count")?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1260
        cx.set(
            "prefix_output",
            cx.method(
                &text("{}output_blocks.{}."),
                "format",
                vec![
                    cx.get("key_prefix")?,
                    cx.binary(
                        "Sub",
                        &cx.binary("Sub", &cx.get("input_block_count")?, &cx.get("count")?)?,
                        &integer(1_i128)?,
                    )?,
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1262
        cx.set(
            "block_keys",
            cx.function(
                "sorted",
                vec![cx.function(
                    "list",
                    vec![cx.comprehension(
                        cx.iter(&cx.get("state_dict_keys")?)?,
                        "list",
                        |cx, item| {
                            cx.set("a", item);
                            if !cx.truth(&cx.method(
                                &cx.get("a")?,
                                "startswith",
                                vec![cx.get("prefix")?],
                                vec![],
                            )?)? {
                                return Ok(None);
                            }
                            Ok(Some(cx.get("a")?))
                        },
                    )?],
                    vec![],
                )?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1263
        if cx.truth(&{
            let detector_tmp_452 = cx.function("len", vec![cx.get("block_keys")?], vec![])?;
            let detector_tmp_453 = integer(0_i128)?;
            cx.compare("Eq", &detector_tmp_452, &detector_tmp_453)?
        })? {
            // comfy/model_detection.py:1264
            break;
        }
        // comfy/model_detection.py:1266
        cx.set(
            "block_keys_output",
            cx.function(
                "sorted",
                vec![cx.function(
                    "list",
                    vec![cx.comprehension(
                        cx.iter(&cx.get("state_dict_keys")?)?,
                        "list",
                        |cx, item| {
                            cx.set("a", item);
                            if !cx.truth(&cx.method(
                                &cx.get("a")?,
                                "startswith",
                                vec![cx.get("prefix_output")?],
                                vec![],
                            )?)? {
                                return Ok(None);
                            }
                            Ok(Some(cx.get("a")?))
                        },
                    )?],
                    vec![],
                )?],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1268
        if cx.truth(&{
            let detector_tmp_454 = cx.method(
                &text("{}0.op.weight"),
                "format",
                vec![cx.get("prefix")?],
                vec![],
            )?;
            let detector_tmp_455 = cx.get("block_keys")?;
            cx.compare("In", &detector_tmp_454, &detector_tmp_455)?
        })? {
            // comfy/model_detection.py:1269
            cx.method(
                &cx.get("num_res_blocks")?,
                "append",
                vec![cx.get("last_res_blocks")?],
                vec![],
            )?;
            // comfy/model_detection.py:1270
            cx.method(
                &cx.get("channel_mult")?,
                "append",
                vec![cx.get("last_channel_mult")?],
                vec![],
            )?;
            // comfy/model_detection.py:1272
            cx.set(
                "current_res",
                cx.binary("Mult", &cx.get("current_res")?, &integer(2_i128)?)?,
            );
            // comfy/model_detection.py:1273
            cx.set("last_res_blocks", integer(0_i128)?);
            // comfy/model_detection.py:1274
            cx.set("last_channel_mult", integer(0_i128)?);
            // comfy/model_detection.py:1275
            cx.set(
                "out",
                cx.function(
                    "calculate_transformer_depth",
                    vec![
                        cx.get("prefix_output")?,
                        cx.get("state_dict_keys")?,
                        cx.get("state_dict")?,
                    ],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:1276
            if cx.truth(&{
                let detector_tmp_456 = cx.get("out")?;
                let detector_tmp_457 = none();
                cx.compare("IsNot", &detector_tmp_456, &detector_tmp_457)?
            })? {
                // comfy/model_detection.py:1277
                cx.method(
                    &cx.get("transformer_depth_output")?,
                    "append",
                    vec![cx.index(&cx.get("out")?, &integer(0_i128)?)?],
                    vec![],
                )?;
            } else {
                // comfy/model_detection.py:1279
                cx.method(
                    &cx.get("transformer_depth_output")?,
                    "append",
                    vec![integer(0_i128)?],
                    vec![],
                )?;
            }
        } else {
            // comfy/model_detection.py:1281
            cx.set(
                "res_block_prefix",
                cx.method(
                    &text("{}0.in_layers.0.weight"),
                    "format",
                    vec![cx.get("prefix")?],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:1282
            if cx.truth(&{
                let detector_tmp_458 = cx.get("res_block_prefix")?;
                let detector_tmp_459 = cx.get("block_keys")?;
                cx.compare("In", &detector_tmp_458, &detector_tmp_459)?
            })? {
                // comfy/model_detection.py:1283
                cx.set(
                    "last_res_blocks",
                    cx.binary("Add", &cx.get("last_res_blocks")?, &integer(1_i128)?)?,
                );
                // comfy/model_detection.py:1284
                cx.set(
                    "last_channel_mult",
                    cx.binary(
                        "FloorDiv",
                        &cx.index(
                            &cx.attribute(
                                &cx.index(
                                    &cx.get("state_dict")?,
                                    &cx.method(
                                        &text("{}0.out_layers.3.weight"),
                                        "format",
                                        vec![cx.get("prefix")?],
                                        vec![],
                                    )?,
                                )?,
                                "shape",
                            )?,
                            &integer(0_i128)?,
                        )?,
                        &cx.get("model_channels")?,
                    )?,
                );
                // comfy/model_detection.py:1286
                cx.set(
                    "out",
                    cx.function(
                        "calculate_transformer_depth",
                        vec![
                            cx.get("prefix")?,
                            cx.get("state_dict_keys")?,
                            cx.get("state_dict")?,
                        ],
                        vec![],
                    )?,
                );
                // comfy/model_detection.py:1287
                if cx.truth(&{
                    let detector_tmp_460 = cx.get("out")?;
                    let detector_tmp_461 = none();
                    cx.compare("IsNot", &detector_tmp_460, &detector_tmp_461)?
                })? {
                    // comfy/model_detection.py:1288
                    cx.method(
                        &cx.get("transformer_depth")?,
                        "append",
                        vec![cx.index(&cx.get("out")?, &integer(0_i128)?)?],
                        vec![],
                    )?;
                    // comfy/model_detection.py:1289
                    if cx.truth(&{
                        let detector_tmp_462 = cx.get("context_dim")?;
                        let detector_tmp_463 = none();
                        cx.compare("Is", &detector_tmp_462, &detector_tmp_463)?
                    })? {
                        // comfy/model_detection.py:1290
                        cx.set("context_dim", cx.index(&cx.get("out")?, &integer(1_i128)?)?);
                        // comfy/model_detection.py:1291
                        cx.set(
                            "use_linear_in_transformer",
                            cx.index(&cx.get("out")?, &integer(2_i128)?)?,
                        );
                        // comfy/model_detection.py:1292
                        cx.set("video_model", cx.index(&cx.get("out")?, &integer(3_i128)?)?);
                        // comfy/model_detection.py:1293
                        cx.set(
                            "video_model_cross",
                            cx.index(&cx.get("out")?, &integer(4_i128)?)?,
                        );
                    }
                } else {
                    // comfy/model_detection.py:1295
                    cx.method(
                        &cx.get("transformer_depth")?,
                        "append",
                        vec![integer(0_i128)?],
                        vec![],
                    )?;
                }
            }
            // comfy/model_detection.py:1297
            cx.set(
                "res_block_prefix",
                cx.method(
                    &text("{}0.in_layers.0.weight"),
                    "format",
                    vec![cx.get("prefix_output")?],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:1298
            if cx.truth(&{
                let detector_tmp_464 = cx.get("res_block_prefix")?;
                let detector_tmp_465 = cx.get("block_keys_output")?;
                cx.compare("In", &detector_tmp_464, &detector_tmp_465)?
            })? {
                // comfy/model_detection.py:1299
                cx.set(
                    "out",
                    cx.function(
                        "calculate_transformer_depth",
                        vec![
                            cx.get("prefix_output")?,
                            cx.get("state_dict_keys")?,
                            cx.get("state_dict")?,
                        ],
                        vec![],
                    )?,
                );
                // comfy/model_detection.py:1300
                if cx.truth(&{
                    let detector_tmp_466 = cx.get("out")?;
                    let detector_tmp_467 = none();
                    cx.compare("IsNot", &detector_tmp_466, &detector_tmp_467)?
                })? {
                    // comfy/model_detection.py:1301
                    cx.method(
                        &cx.get("transformer_depth_output")?,
                        "append",
                        vec![cx.index(&cx.get("out")?, &integer(0_i128)?)?],
                        vec![],
                    )?;
                } else {
                    // comfy/model_detection.py:1303
                    cx.method(
                        &cx.get("transformer_depth_output")?,
                        "append",
                        vec![integer(0_i128)?],
                        vec![],
                    )?;
                }
            }
        }
    }
    // comfy/model_detection.py:1306
    cx.method(
        &cx.get("num_res_blocks")?,
        "append",
        vec![cx.get("last_res_blocks")?],
        vec![],
    )?;
    // comfy/model_detection.py:1307
    cx.method(
        &cx.get("channel_mult")?,
        "append",
        vec![cx.get("last_channel_mult")?],
        vec![],
    )?;
    // comfy/model_detection.py:1308
    if cx.truth(&{
        let detector_tmp_468 = cx.method(
            &text("{}middle_block.1.proj_in.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?;
        let detector_tmp_469 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_468, &detector_tmp_469)?
    })? {
        // comfy/model_detection.py:1309
        cx.set(
            "transformer_depth_middle",
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict_keys")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("{}middle_block.1.transformer_blocks."),
                            "format",
                            vec![cx.get("key_prefix")?],
                            vec![],
                        )?,
                        &text("{}"),
                    )?,
                ],
                vec![],
            )?,
        );
    } else {
        // comfy/model_detection.py:1310
        if cx.truth(&{
            let detector_tmp_470 = cx.method(
                &text("{}middle_block.0.in_layers.0.weight"),
                "format",
                vec![cx.get("key_prefix")?],
                vec![],
            )?;
            let detector_tmp_471 = cx.get("state_dict_keys")?;
            cx.compare("In", &detector_tmp_470, &detector_tmp_471)?
        })? {
            // comfy/model_detection.py:1311
            cx.set(
                "transformer_depth_middle",
                cx.unary("USub", &integer(1_i128)?)?,
            );
        } else {
            // comfy/model_detection.py:1313
            cx.set(
                "transformer_depth_middle",
                cx.unary("USub", &integer(2_i128)?)?,
            );
        }
    }
    // comfy/model_detection.py:1315
    cx.set_item(
        &cx.get("unet_config")?,
        &text("in_channels"),
        cx.get("in_channels")?,
    )?;
    // comfy/model_detection.py:1316
    cx.set_item(
        &cx.get("unet_config")?,
        &text("out_channels"),
        cx.get("out_channels")?,
    )?;
    // comfy/model_detection.py:1317
    cx.set_item(
        &cx.get("unet_config")?,
        &text("model_channels"),
        cx.get("model_channels")?,
    )?;
    // comfy/model_detection.py:1318
    cx.set_item(
        &cx.get("unet_config")?,
        &text("num_res_blocks"),
        cx.get("num_res_blocks")?,
    )?;
    // comfy/model_detection.py:1319
    cx.set_item(
        &cx.get("unet_config")?,
        &text("transformer_depth"),
        cx.get("transformer_depth")?,
    )?;
    // comfy/model_detection.py:1320
    cx.set_item(
        &cx.get("unet_config")?,
        &text("transformer_depth_output"),
        cx.get("transformer_depth_output")?,
    )?;
    // comfy/model_detection.py:1321
    cx.set_item(
        &cx.get("unet_config")?,
        &text("channel_mult"),
        cx.get("channel_mult")?,
    )?;
    // comfy/model_detection.py:1322
    cx.set_item(
        &cx.get("unet_config")?,
        &text("transformer_depth_middle"),
        cx.get("transformer_depth_middle")?,
    )?;
    // comfy/model_detection.py:1323
    cx.set_item(
        &cx.get("unet_config")?,
        &text("use_linear_in_transformer"),
        cx.get("use_linear_in_transformer")?,
    )?;
    // comfy/model_detection.py:1324
    cx.set_item(
        &cx.get("unet_config")?,
        &text("context_dim"),
        cx.get("context_dim")?,
    )?;
    // comfy/model_detection.py:1326
    if cx.truth(&cx.get("video_model")?)? {
        // comfy/model_detection.py:1327
        cx.set_item(
            &cx.get("unet_config")?,
            &text("extra_ff_mix_layer"),
            boolean(true),
        )?;
        // comfy/model_detection.py:1328
        cx.set_item(
            &cx.get("unet_config")?,
            &text("use_spatial_context"),
            boolean(true),
        )?;
        // comfy/model_detection.py:1329
        cx.set_item(
            &cx.get("unet_config")?,
            &text("merge_strategy"),
            text("learned_with_images"),
        )?;
        // comfy/model_detection.py:1330
        cx.set_item(
            &cx.get("unet_config")?,
            &text("merge_factor"),
            float(0.0_f64),
        )?;
        // comfy/model_detection.py:1331
        cx.set_item(
            &cx.get("unet_config")?,
            &text("video_kernel_size"),
            list(vec![integer(3_i128)?, integer(1_i128)?, integer(1_i128)?]),
        )?;
        // comfy/model_detection.py:1332
        cx.set_item(
            &cx.get("unet_config")?,
            &text("use_temporal_resblock"),
            boolean(true),
        )?;
        // comfy/model_detection.py:1333
        cx.set_item(
            &cx.get("unet_config")?,
            &text("use_temporal_attention"),
            boolean(true),
        )?;
        // comfy/model_detection.py:1334
        cx.set_item(
            &cx.get("unet_config")?,
            &text("disable_temporal_crossattention"),
            cx.unary("Not", &cx.get("video_model_cross")?)?,
        )?;
    } else {
        // comfy/model_detection.py:1336
        cx.set_item(
            &cx.get("unet_config")?,
            &text("use_temporal_resblock"),
            boolean(false),
        )?;
        // comfy/model_detection.py:1337
        cx.set_item(
            &cx.get("unet_config")?,
            &text("use_temporal_attention"),
            boolean(false),
        )?;
    }
    // comfy/model_detection.py:1339
    cx.set(
        "heatmap_key",
        cx.method(
            &text("{}heatmap_head.conv_layers.0.weight"),
            "format",
            vec![cx.get("key_prefix")?],
            vec![],
        )?,
    );
    // comfy/model_detection.py:1340
    if cx.truth(&{
        let detector_tmp_472 = cx.get("heatmap_key")?;
        let detector_tmp_473 = cx.get("state_dict_keys")?;
        cx.compare("In", &detector_tmp_472, &detector_tmp_473)?
    })? {
        // comfy/model_detection.py:1341
        cx.set_item(
            &cx.get("unet_config")?,
            &text("heatmap_head"),
            boolean(true),
        )?;
    }
    cx.get("unet_config")
}
